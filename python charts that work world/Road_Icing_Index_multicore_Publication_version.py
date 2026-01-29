###########################################
# WRF Road Icing Index Calculation Script #
###########################################
#
# This script calculates a Road Icing Index (RII) from WRF model output. The RII is
# a conceptual metric designed to highlight regions with increased potential
# for road icing conditions, integrating temperature, wet-bulb temperature,
# precipitation type and rate, humidity, and wind speed.
#
# The approach implemented here draws upon research in transportation meteorology,
# including:
#
# Gustafsson, D., Bogren, J., & Greenfield, T. (2010). Road Weather Information Systems.
# IEEE Instrumentation & Measurement Magazine, 13(2), 28–32. https://doi.org/10.1109/MIM.2010.5448478
#
# Strategic Highway Research Program (SHRP 2) Report S2-L01-RR-1:
# "Integrating Mobile Observations into Winter Road Maintenance Decisions" (2012).
# Transportation Research Board. http://onlinepubs.trb.org/onlinepubs/shrp2/SHRP2prepubL01Report.pdf
#
# NOAA Road Weather Management Program:
#   https://ops.fhwa.dot.gov/weather/
#
# Rauber, R. M., et al. (2001). "Precipitation Processes in Winter Weather Systems."
# Meteorological Monographs, 28(50), 3-15. https://doi.org/10.1175/0065-9401-28.50.3
#
# Chapman, L., & Thornes, J. E. (2011). "What Makes Roads Slick? A Review of Surface Ice Prediction Systems."
# Quarterly Journal of the Royal Meteorological Society, 137(659), 15–31. https://doi.org/10.1002/qj.758
#
# Karsisto, V., Nurmi, P., Fortelius, C., & Kangas, M. (2017).
# "Improving Road Weather Model Forecasts with Statistical Post-Processing Techniques."
# Meteorological Applications, 24(2), 169–178. https://doi.org/10.1002/met.1624
#
# Skamarock, W. C., Klemp, J. B., Dudhia, J., Gill, D. O., Barker, D. M., Wang, W., & Powers, J. G. (2008).
# "A Description of the Advanced Research WRF Version 3."
# NCAR Technical Note NCAR/TN-475+STR. https://doi.org/10.5065/D68S4MVH
#
# World Meteorological Organization. (2014).
# Guide to meteorological instruments and methods of observation (WMO-No. 8).
# World Meteorological Organization.
#
# Methodology and Data Sources:
# - WRF-ARW Model Output: We use WRF-Python (https://wrf-python.readthedocs.io/en/latest/)
#   to extract model fields and compute derived diagnostics.
# - Meteorological calculations (wet-bulb temperature, dew point) performed using MetPy:
#   https://unidata.github.io/MetPy/latest/
#
# Precipitation and Temperature Profile:
# The precipitation type logic (here replaced with a hybrid approach) draws upon:
# - Bourgouin, P. (2000). A Method to Determine Precipitation Types. Weather and Forecasting, 15, 583–592.
# - Ramer, J. (1993). An On-Line Diagnostic Precipitation Type Algorithm for Winter Weather.
#   5th Intl. Conf. on Aviation Weather Systems, 227–230.
# - Baldwin, M. E., Contorno, L. J., & Didlake, A. C. (1994). An Algorithm for Forecasting Precipitation Type.
#   Meteorological Applications.
# - Ensemble/Hybrid Approaches: Reeves, H. D., et al. (2014), plus additional references therein.
#
# In this version, we extend the vertical slice up to ~700 mb instead of ~850 mb
# to account for deeper potential warm layers aloft (Rauber et al., 2001).
#
# Assumptions for Road Icing Index:
# - Temperature thresholds for icing risk are based on operational experience and
#   published guidelines (e.g., Gustafsson et al., 2010).
# - The weighting of precipitation type and rate in the RII is a heuristic approach
#   reflecting the increased risk of icing when freezing precipitation occurs (cf. SHRP2).
# - The wet-bulb temperature threshold is used because it represents the temperature
#   at which evaporation and cooling can facilitate icing (Reeves et al., 2014).
# - Relative humidity and wind speed factors are included to represent rapid evaporative
#   cooling and heat flux from the road surface, drawing conceptually from studies
#   in road surface energy budgets (e.g., Chapman & Thornes, 2011, QJRMS).
#
# Precipitation Rate Calculation:
# Precipitation rates are derived from cumulative fields RAINC, RAINNC, RAINSH, and SNOW:
# - RAINC/RAINNC/RAINSH/SNOW: WRF cumulative precipitation fields (Skamarock et al., 2008, NCAR/TN–475+STR).
#
# The one-hour differences of cumulative precipitation fields are taken to estimate
# precipitation intensity in mm/h, a common approach in meteorological post-processing.
#
# This script is intended as a conceptual example and may require adaptation or calibration
# for operational use.

# --- Standard Libraries ---
import os
import sys
from datetime import datetime
import glob
import re
from concurrent.futures import ProcessPoolExecutor

# --- Third-Party Libraries ---
import numpy as np
import numpy.ma as ma
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.cm import get_cmap
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from PIL import Image
import metpy.calc as mpcalc
from metpy.calc import dewpoint_from_relative_humidity, wet_bulb_temperature
from metpy.units import units
import metpy
import xarray as xr
import geopandas as gpd
import warnings

# --- WRF-Python Libraries ---
import wrf
from wrf import getvar, interplevel, to_np, ALL_TIMES

# Ignore unnecessary warnings for cleaner output
warnings.filterwarnings("ignore")

######################################
# Command-line Argument Verification #
######################################
if len(sys.argv) != 3:
    print(
        "\nEnter the required arguments: path_wrf and domain\n"
        "For example: script_name.py /home/WRF/test/em_real d01\n"
    )
    sys.exit()

path_wrf = sys.argv[1]
domain = sys.argv[2]

#######################################
# Output Directories for Results      #
#######################################
output_folder = "wrf_Road_Icing_Index"
image_folder = os.path.join(output_folder, "Images")
animation_folder = os.path.join(output_folder, "Animation")

for folder in [output_folder, image_folder, animation_folder]:
    os.makedirs(folder, exist_ok=True)


###############################################
# Map Feature Addition Function (Cartopy)     #
###############################################
def add_feature(
    ax, category, scale, facecolor, edgecolor, linewidth, name, zorder=None, alpha=None
):
    """
    Add a Natural Earth feature to the map axis.

    References:
    Natural Earth Data: https://www.naturalearthdata.com/

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The map axis.
    category : str
        NE category (e.g., "physical", "cultural").
    scale : str
        Data scale, e.g., "10m".
    facecolor : str or None
        Face color of the feature.
    edgecolor : str or None
        Edge color of the feature.
    linewidth : float or None
        Line width of feature edges.
    name : str
        Name of the NE feature (e.g., "coastline").
    zorder : int or None
        Z-order for layering features.
    alpha : float or None
        Transparency level.
    """
    feature = cfeature.NaturalEarthFeature(
        category=category,
        scale=scale,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        name=name,
        zorder=zorder,
        alpha=alpha,
    )
    ax.add_feature(feature)


#####################################
# List of Features to Add on the Map #
#####################################
# List of features to add
map_features = [
    ("physical", "10m", cfeature.COLORS["land"], "black", 0.50, "minor_islands"),
    ("physical", "10m", "none", "black", 0.50, "coastline"),
    ("physical", "10m", cfeature.COLORS["water"], None, None, "ocean_scale_rank", -1),
    ("physical", "10m", cfeature.COLORS["water"], "lightgrey", 0.75, "lakes", 0),
    ("cultural", "10m", "none", "grey", 0.75, "admin_1_states_provinces", 2),
    ("cultural", "10m", "none", "black", 1.30, "admin_0_countries", 2),
    ("cultural", "10m", "none", "black", 0.60, "admin_2_counties", 2, 0.6),
    ("cultural", "10m", "none", "red", 0.80, "roads", 2),
    ("cultural", "10m", "none", "red", 0.80, "roads_north_america", 2),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_lake_centerlines"),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_north_america", None), 0.75),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_australia", None), 0.75),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_europe", None), 0.75),
    # ("physical", "10m", cfeature.COLORS["water"], cfeature.COLORS["water"], None, "lakes_north_america", None), 0.75),
    # ("physical", "10m", cfeature.COLORS["water"], cfeature.COLORS["water"], None, "lakes_australia", None), 0.75),
    # ("physical", "10m", cfeature.COLORS["water"], cfeature.COLORS["water"], None, "lakes_europe", None), 0.75)]
]

################################
# Load Cities Data from URL    #
################################
# Source: Natural Earth Populated Places
# https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


##################################################################################
# Hybrid Precipitation Type Calculation Function (Bourgouin, Ramer, Baldwin, etc.)
##################################################################################
def calculate_hybrid_precipitation_type(
    temp_3d, pressure_3d, t2_da, psfc_da, qrain, qsnow, qgraup, wet_bulb_temp
):
    """
    Determine precipitation type using a hybrid method combining key aspects of the
    Bourgouin, Ramer, and Baldwin algorithms, plus an ensemble approach for final classification.

    This function examines vertical temperature profiles up to ~700 mb, layer thicknesses,
    and surface mixing ratios (rain, snow, graupel) to classify precipitation types
    such as Rain, Freezing Rain, Snow, Sleet/Graupel, or None.

    References:
    - Bourgouin, P. (2000). A Method to Determine Precipitation Types. Weather and Forecasting, 15, 583–592.
    - Ramer, J. (1993). An On-Line Diagnostic Precipitation Type Algorithm for Winter Weather.
      5th Intl. Conf. on Aviation Weather Systems, 227–230.
    - Baldwin, M. E., et al. (1994). An Algorithm for Forecasting Precipitation Type. Meteorological Applications.
    - Rauber, R. M., et al. (2001). "Precipitation Processes in Winter Weather Systems."
      Meteorological Monographs, 28(50), 3–15. https://doi.org/10.1175/0065-9401-28.50.3
    - Chapman, L., & Thornes, J. E. (2011). "What Makes Roads Slick? A Review of Surface Ice Prediction Systems."
      Quarterly Journal of the Royal Meteorological Society, 137(659), 15–31.
    - Reeves, H. D., et al. (2014). "Adding Value to WRF-Generated Probabilistic Guidance for Cool-Season Severe Weather."
      Weather and Forecasting, 29(6), 1255–1277.

    Parameters
    ----------
    temp_3d : xarray.DataArray
        3D temperature in Kelvin (perturbation potential temperature already converted to real T).
    pressure_3d : xarray.DataArray
        3D pressure in Pa (perturbation + base state).
    t2_da : xarray.DataArray
        2 m temperature (K).
    psfc_da : xarray.DataArray
        Surface pressure (Pa).
    qrain : xarray.DataArray
        3D rain mixing ratio (kg/kg).
    qsnow : xarray.DataArray
        3D snow mixing ratio (kg/kg).
    qgraup : xarray.DataArray
        3D graupel mixing ratio (kg/kg).
    wet_bulb_temp : ndarray
        2D wet-bulb temperature at the surface (K).

    Returns
    -------
    ptype : ndarray (2D)
        Precipitation type classification array (strings):
        ['Rain', 'Freezing Rain', 'Snow', 'Sleet/Graupel', 'None']

    Notes
    -----
    - Truncates the temperature profile near ~700 mb instead of 850 mb, capturing deeper warm layers
      that could influence freezing rain/sleet formation (Rauber et al., 2001).
    - Uses ensemble-like voting from multiple methods (Bourgouin, Ramer, Baldwin).
    - Resolves ambiguous classification with wet-bulb temperature checks and mixing ratios at surface.
    """
    # Convert near-surface temperature to Celsius (# Gustafsson et al., 2010 for threshold logic)
    t2_c = t2_da.values - 273.15
    wb_c = wet_bulb_temp - 273.15  # 2D array of wet-bulb in °C

    # Truncate temperature profile to ~700 mb (Rauber et al., 2001)
    ref_pressure_profile = pressure_3d.isel(south_north=0, west_east=0).values
    idx_700 = np.argmin(np.abs(ref_pressure_profile - 700.0))  # find 700 mb
    temp_profile_700 = temp_3d.isel(bottom_top=slice(0, idx_700 + 1))

    # Extract surface-level mixing ratios
    qrain_surf = qrain.isel(bottom_top=0).values
    qsnow_surf = qsnow.isel(bottom_top=0).values
    qgraup_surf = qgraup.isel(bottom_top=0).values

    # Convert truncated profile to Celsius
    temp_c_profile = temp_profile_700.values - 273.15  # shape: [levels, y, x]

    # Identify any layer above freezing vs. entire below freezing (Bourgouin approach)
    above_freezing_any = np.any(temp_c_profile > 0, axis=0)
    below_freezing_all = np.all(temp_c_profile <= 0, axis=0)

    # Initialize precipitation type array
    ptype_hybrid = np.full(t2_da.shape, "None", dtype=object)

    # ----------------------------------------------------------
    # 1) Preliminary "Raw" Classifications from Microphysics
    # ----------------------------------------------------------
    # If there's a significant mixing ratio of rain, snow, or graupel at surface level
    is_rain_surf = qrain_surf > 1e-6
    is_snow_surf = qsnow_surf > 1e-6
    is_graup_surf = qgraup_surf > 1e-6

    # ----------------------------------------------------------
    # 2) Ramer/Baldwin Criteria (simplified)
    # ----------------------------------------------------------
    # Ramer-like: if temperature above 0°C in upper slices but <=0°C at surface => freezing rain
    # Baldwin-like logic can also consider thickness/delta T, but here simplified
    level_temp_high = temp_profile_700.isel(bottom_top=-1).values - 273.15
    surface_temp_c = t2_c

    ramer_freezing_rain = (level_temp_high > 0) & (surface_temp_c <= 0)
    ramer_snow = surface_temp_c <= 0
    ramer_rain = surface_temp_c > 0

    # ----------------------------------------------------------
    # 3) Bourgouin Warm/Cold Layer Detection (Simplified)
    # ----------------------------------------------------------
    # Warming layer above, subfreezing at surface => freezing rain or sleet
    warm_layer_bourgouin = above_freezing_any
    cold_surface_bourgouin = surface_temp_c <= 0
    bourgouin_freezing_rain = warm_layer_bourgouin & cold_surface_bourgouin

    # ----------------------------------------------------------
    # 4) Ensemble Voting
    # ----------------------------------------------------------
    votes_rain = np.zeros_like(t2_da.values, dtype=int)
    votes_freezing_rain = np.zeros_like(t2_da.values, dtype=int)
    votes_snow = np.zeros_like(t2_da.values, dtype=int)
    votes_sleet = np.zeros_like(t2_da.values, dtype=int)

    # Based on microphysics presence
    votes_snow[is_snow_surf] += 1
    votes_sleet[is_graup_surf] += 1
    votes_rain[is_rain_surf & (surface_temp_c > 0)] += 1
    votes_freezing_rain[is_rain_surf & (surface_temp_c <= 0)] += 1

    # Based on Ramer approach
    votes_freezing_rain[ramer_freezing_rain] += 1
    votes_snow[ramer_snow] += 1
    votes_rain[ramer_rain] += 1

    # Based on Bourgouin
    votes_freezing_rain[bourgouin_freezing_rain] += 1

    # Decide final classification by majority (Reeves et al., 2014 for ensemble)
    ptype_hybrid[votes_freezing_rain >= 2] = "Freezing Rain"
    ptype_hybrid[votes_snow >= 2] = "Snow"
    ptype_hybrid[votes_sleet >= 2] = "Sleet/Graupel"
    ptype_hybrid[votes_rain >= 2] = "Rain"

    # ----------------------------------------------------------
    # 5) Resolve Ambiguous Grid Cells with Wet-Bulb Temperature
    # ----------------------------------------------------------
    ambiguous_mask = ptype_hybrid == "None"
    ptype_hybrid[ambiguous_mask & (wb_c < 0)] = "Freezing Rain"
    ptype_hybrid[ambiguous_mask & (wb_c >= 0) & (wb_c <= 2)] = "Rain"
    ptype_hybrid[ambiguous_mask & (wb_c > 2)] = "Rain"

    # If no surface microphysics present, set to "None"
    no_precip_mask = ~is_rain_surf & ~is_snow_surf & ~is_graup_surf
    ptype_hybrid[no_precip_mask] = "None"

    return ptype_hybrid


###################################################################
# Vectorized Function to Calculate Road Icing Index (RII)
###################################################################
def calculate_icing_index_vectorized(
    t2k, wbk, ptype, precip_rate, rh, ws, top_soil_temp_k
):
    """
    Compute Road Icing Index (RII) in a vectorized manner.

    Parameters
    ----------
    t2k : ndarray
        Near-surface air temperature in Kelvin.
    wbk : ndarray
        Wet-bulb temperature in Kelvin.
    ptype : ndarray
        Precipitation type classification (e.g., 'Rain', 'Snow', 'Freezing Rain', 'Sleet/Graupel', 'None').
    precip_rate : ndarray
        Precipitation rate in mm/h.
    rh : ndarray
        Relative humidity in %.
    ws : ndarray
        Wind speed in m/s.
    top_soil_temp_k : ndarray
        Top-layer soil temperature in Kelvin.

    Returns
    -------
    icing_index : ndarray
        Calculated Road Icing Index for each grid cell.

    References
    ----------
    - Gustafsson, D., Bogren, J., & Greenfield, T. (2010). "Road Weather Information Systems."
      IEEE Instrumentation & Measurement Magazine, 13(2), 28–32. https://doi.org/10.1109/MIM.2010.5448478
    - SHRP2 Report S2-L01-RR-1: "Integrating Mobile Observations into Winter Road Maintenance Decisions." (2012).
      Transportation Research Board. http://onlinepubs.trb.org/onlinepubs/shrp2/SHRP2prepubL01Report.pdf
    - Rauber, R. M., et al. (2001). "Precipitation Processes in Winter Weather Systems."
      Meteorological Monographs, 28(50), 3-15. https://doi.org/10.1175/0065-9401-28.50.3
    - Chapman, L., & Thornes, J. E. (2011). "What Makes Roads Slick? A Review of Surface Ice Prediction Systems."
      Quarterly Journal of the Royal Meteorological Society, 137(659), 15–31. https://doi.org/10.1002/qj.758
    - Karsisto, V., Nurmi, P., Fortelius, C., & Kangas, M. (2017). "Improving Road Weather Model Forecasts with
      Statistical Post-Processing Techniques." Meteorological Applications, 24(2), 169–178.
    """
    # Convert temperatures from Kelvin to Celsius
    t2_c = t2k - 273.15
    wb_c = wbk - 273.15
    soil_c = top_soil_temp_k - 273.15

    # Temperature Score (Gustafsson et al., 2010)
    temp_score = np.zeros_like(t2_c)
    temp_score[t2_c < -5] = 5  # 3
    temp_score[(t2_c >= -5) & (t2_c <= 0)] = 3  # 2
    temp_score[(t2_c > 0) & (t2_c <= 3)] = 2
    temp_score[(t2_c > 3) & (t2_c <= 5)] = -1  # agregada por mi
    temp_score[(t2_c > 5) & (t2_c <= 10)] = -3  # agregada por mi
    temp_score[t2_c > 10] = -5  # agregada por mi

    # Wet-Bulb Temperature Score (Chapman & Thornes, 2011)
    wet_bulb_score = np.zeros_like(wb_c)
    wet_bulb_score[wb_c < 0] = 2
    wet_bulb_score[(wb_c >= 0) & (wb_c <= 2)] = 1

    # Precipitation Type Score (Rauber et al., 2001, SHRP2)
    ptype_score_map = {
        "Rain": 1,  # Originalmete estaba en 1, lo cambio a 0 para probar <--------------------------------------------------------------------------------------********************************************
        "Freezing Rain": 3,
        "Snow": 2,
        "Sleet/Graupel": 2,
        "None": 0,
    }
    ptype_score = np.zeros_like(t2_c)
    for pt, val in ptype_score_map.items():
        ptype_score[ptype == pt] = val

    # Precipitation Rate Score
    precip_rate_score = np.zeros_like(precip_rate)

    # Masks for precipitation type
    freezing_rain_mask = ptype == "Freezing Rain"
    snow_mask = (ptype == "Snow") | (ptype == "Sleet/Graupel")
    rain_mask = ptype == "Rain"

    # Freezing Rain thresholds (SHRP2, 2012)
    precip_rate_score[freezing_rain_mask & (precip_rate <= 2.5)] = 1
    precip_rate_score[
        freezing_rain_mask & (precip_rate > 2.5) & (precip_rate <= 7.5)
    ] = 2
    precip_rate_score[freezing_rain_mask & (precip_rate > 7.5)] = 3

    # Snow thresholds (liquid equivalent) (Gustafsson et al., 2010)
    precip_rate_score[snow_mask & (precip_rate <= 1.0)] = 1
    precip_rate_score[snow_mask & (precip_rate > 1.0) & (precip_rate <= 3.0)] = 2
    precip_rate_score[snow_mask & (precip_rate > 3.0)] = 3

    # Rain thresholds (Rauber et al., 2001)
    precip_rate_score[rain_mask & (precip_rate <= 5.0)] = 1
    precip_rate_score[rain_mask & (precip_rate > 5.0) & (precip_rate <= 15.0)] = 2  # 2
    precip_rate_score[rain_mask & (precip_rate > 15.0)] = 3  # 3

    # Relative Humidity Score (Chapman & Thornes, 2011)
    rh_score = np.zeros_like(rh)
    rh_score[rh >= 90] = 1

    # Wind Speed Score (Gustafsson et al., 2010; SHRP2, 2012)
    wind_score = np.zeros_like(ws)
    wind_score[ws < 10] = 1
    wind_score[(ws >= 10) & (ws <= 20)] = 0
    wind_score[ws > 20] = -1

    # Soil Temperature Score (Karsisto et al., 2017)
    soil_score = np.zeros_like(soil_c)
    soil_score[soil_c < 0] = 1

    # Summation for Road Icing Index
    #   Each factor is added, reflecting the relative contribution to icing risk.
    icing_index = (
        temp_score
        + wet_bulb_score
        + ptype_score
        + precip_rate_score
        + rh_score
        + wind_score
        + soil_score
    )

    return icing_index


###########################################
# Prepare Variables for Precipitation Rate #
###########################################
prev_total_rain = None
prev_total_snow = None
prev_datetime = None


###############################################################################
# Process a single WRF output file (for multiprocessing)
###############################################################################
def process_wrf_file(args):
    index, wrf_file_path, prev_wrf_file_path, domain, image_folder = args

    # Open WRF output file
    wrf_file = Dataset(wrf_file_path)
    ds = xr.open_dataset(wrf_file_path)

    # Extract date/time from filename
    year = wrf_file_path[
        wrf_file_path.find("wrfout") + 11 : wrf_file_path.find("wrfout") + 15
    ]
    month = wrf_file_path[
        wrf_file_path.find("wrfout") + 16 : wrf_file_path.find("wrfout") + 18
    ]
    day = wrf_file_path[
        wrf_file_path.find("wrfout") + 19 : wrf_file_path.find("wrfout") + 21
    ]
    hour = wrf_file_path[
        wrf_file_path.find("wrfout") + 22 : wrf_file_path.find("wrfout") + 24
    ]
    minute = wrf_file_path[
        wrf_file_path.find("wrfout") + 25 : wrf_file_path.find("wrfout") + 27
    ]

    print(f"Plotting data: {year}/{month}/{day} {hour}:{minute} UTC")

    ############################################
    # Extracting Key Variables from WRF Model #
    ############################################

    # 2-Meter Temperature (T2)
    t2_da = getvar(wrf_file, "T2")

    # Relative Humidity at 2 meters (RH2)
    rh_da = getvar(wrf_file, "rh2")

    # U/V Components at 10 meters
    u10_da = getvar(wrf_file, "U10")
    v10_da = getvar(wrf_file, "V10")

    # 3D Rain, Snow, Graupel Mixing Ratios
    qrain_3d = getvar(wrf_file, "QRAIN")
    qsnow_3d = getvar(wrf_file, "QSNOW")
    qgraup_3d = getvar(wrf_file, "QGRAUP")

    # Surface Pressure
    psfc_da = getvar(wrf_file, "PSFC")

    # Soil Temperature (top layer)
    tslb_da = getvar(wrf_file, "TSLB")
    top_soil_temp = tslb_da.isel(soil_layers_stag=0)

    # Cumulative Precip Fields
    rainc = getvar(wrf_file, "RAINC")
    rainnc = getvar(wrf_file, "RAINNC")
    rainsh = getvar(wrf_file, "RAINSH")
    snow_accum = getvar(wrf_file, "SNOW")  # in kg/m^2

    # 3D Temperature and Pressure
    temp_3d = getvar(wrf_file, "temp", units="K")  # Full temperature field
    pressure_3d = getvar(wrf_file, "pressure")  # Full pressure field

    # Latitude/Longitude
    lats, lons = wrf.latlon_coords(psfc_da)

    ###########################################
    # Calculate Precipitation Rate (mm/h)     #
    ###########################################
    # Current cumulative totals
    total_rain = rainc + rainnc + rainsh  # mm
    total_snow_mm = snow_accum  # kg/m^2 => 1 mm water eq. = 1 kg/m^2

    # Previous cumulative totals from previous file (if provided)
    if prev_wrf_file_path is not None:
        prev_wrf_file = Dataset(prev_wrf_file_path)
        prev_rainc = getvar(prev_wrf_file, "RAINC")
        prev_rainnc = getvar(prev_wrf_file, "RAINNC")
        prev_rainsh = getvar(prev_wrf_file, "RAINSH")
        prev_snow_accum = getvar(prev_wrf_file, "SNOW")
        prev_total_rain = (prev_rainc + prev_rainnc + prev_rainsh).values
        prev_total_snow = prev_snow_accum.values
        prev_wrf_file.close()
    else:
        prev_total_rain = None
        prev_total_snow = None

    if prev_total_rain is not None:
        one_hour_rain = total_rain.values - prev_total_rain
    else:
        one_hour_rain = np.zeros_like(total_rain.values)

    if prev_total_snow is not None:
        one_hour_snow = total_snow_mm.values - prev_total_snow
    else:
        one_hour_snow = np.zeros_like(total_snow_mm.values)

    precip_rate_arr = one_hour_rain + one_hour_snow  # mm/h

    ########################################
    # Dew Point and Wet-Bulb Calculations  #
    ########################################
    t2_c = t2_da.metpy.convert_units("degC").values * units.degC
    rh_vals = rh_da.values * units.percent
    td2_c = dewpoint_from_relative_humidity(t2_c, rh_vals)  # dewpoint in °C
    p_hPa = psfc_da.metpy.convert_units("hPa").values * units.hPa
    twb_c = wet_bulb_temperature(p_hPa, t2_c, td2_c)  # wet-bulb in °C
    wet_bulb_temp_2d = twb_c.to("K").m  # convert to Kelvin

    ########################################
    # Wind Speed
    ########################################
    wind_speed = np.sqrt(u10_da.values**2 + v10_da.values**2)

    ########################################
    # Hybrid Precipitation Type
    ########################################
    ptype_hybrid = calculate_hybrid_precipitation_type(
        temp_3d=temp_3d,
        pressure_3d=pressure_3d,
        t2_da=t2_da,
        psfc_da=psfc_da,
        qrain=qrain_3d,
        qsnow=qsnow_3d,
        qgraup=qgraup_3d,
        wet_bulb_temp=wet_bulb_temp_2d,
    )

    ########################################
    # Road Icing Index Calculation
    ########################################
    icing_index_arr = calculate_icing_index_vectorized(
        t2k=t2_da.values,
        wbk=wet_bulb_temp_2d,
        ptype=ptype_hybrid,
        precip_rate=precip_rate_arr,
        rh=rh_da.values,
        ws=wind_speed,
        top_soil_temp_k=top_soil_temp.values,
    )

    # Smooth the RII field for aesthetics
    icing_index_arr_smoothed = gaussian_filter(icing_index_arr, sigma=0.5)

    ########################################
    # Retrieve Projection and Grid Spacing
    ########################################
    cart_proj = wrf.get_cartopy(psfc_da)
    lats_grid = ds["XLAT"].metpy.unit_array
    lons_grid = ds["XLONG"].metpy.unit_array
    dx, dy = mpcalc.lat_lon_grid_deltas(lons_grid, lats_grid)
    dx_km = np.round(dx.to(units.kilometer).magnitude)
    dy_km = np.round(dy.to(units.kilometer).magnitude)
    avg_dx_km = round(np.mean(dx_km), 2)
    avg_dy_km = round(np.mean(dy_km), 2)

    ########################################
    # Plotting
    ########################################
    fig = plt.figure(figsize=(19.2, 10.8), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

    # Adjust map extent
    extent_adjustment = (
        0.50 if avg_dx_km >= 9 else (0.25 if 3 < avg_dx_km < 9 else 0.15)
    )
    ax.set_extent(
        [
            lons.min() - extent_adjustment,
            lons.max() + extent_adjustment,
            lats.min() - extent_adjustment,
            lats.max() + extent_adjustment,
        ],
        crs=ccrs.PlateCarree(),
    )

    # Add base land
    ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])
    for feature in map_features:
        add_feature(ax, *feature)

    # Filter and plot cities
    plot_extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    cities_in_extent = cities.cx[
        plot_extent[0] : plot_extent[1], plot_extent[2] : plot_extent[3]
    ]
    sorted_cities = cities_in_extent.sort_values(by="POP_MAX", ascending=False).head(
        100
    )

    # Heuristic minimum distance to avoid overlap
    min_distance = 1.0 if avg_dx_km >= 9 else (0.75 if 3 < avg_dx_km < 9 else 0.40)

    filtered_cities = gpd.GeoDataFrame(columns=sorted_cities.columns)
    for _, city in sorted_cities.iterrows():
        if not filtered_cities.empty:
            distances = filtered_cities.geometry.distance(city.geometry)
            if distances.min() >= min_distance:
                filtered_cities = filtered_cities.append(city, ignore_index=True)
        else:
            filtered_cities = filtered_cities.append(city, ignore_index=True)

    for _, city in filtered_cities.iterrows():
        ax.plot(
            city.geometry.x,
            city.geometry.y,
            marker="o",
            markersize=3,
            color="r",
            clip_on=True,
            transform=ccrs.PlateCarree(),
        )
        ax.text(
            city.geometry.x,
            city.geometry.y,
            city.NAME,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", alpha=0.4),
        )

    # Add gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linestyle="--", alpha=0.5
    )
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ####################################################
    # Road Icing Index Color Table (Minimal to Severe)
    ####################################################
    roadicing_color_table = (
        np.array(
            [
                (0, 255, 255),  # Cyan (Minimal Risk)
                (0, 255, 0),  # Green (Low Risk)
                (255, 255, 0),  # Yellow (Moderate Risk)
                (255, 165, 0),  # Orange (High Risk)
                (255, 0, 0),  # Red (Severe Risk)
            ],
            np.float32,
        )
        / 255.0
    )
    levels = np.array([0, 3, 6, 9, 12, 15])
    roadicing_map = plt.cm.colors.ListedColormap(roadicing_color_table)
    roadicing_norm = plt.cm.colors.BoundaryNorm(levels, roadicing_map.N)

    # Contourf
    roadicing_contourf = ax.contourf(
        to_np(lons),
        to_np(lats),
        icing_index_arr_smoothed,
        levels=levels,
        cmap=roadicing_map,
        norm=roadicing_norm,
        extend="max",
        zorder=0,
        transform=ccrs.PlateCarree(),
    )

    # Colorbar
    cbar = plt.colorbar(
        roadicing_contourf,
        ax=ax,
        orientation="vertical",
        pad=0.05,
        shrink=0.8,
        ticks=levels,
    )
    tick_positions = levels[:-1]
    icing_labels = [
        "Minimal Risk (0-3)",
        "Low Risk (3-6)",
        "Moderate Risk (6-9)",
        "High Risk (9-12)",
        "Severe Risk (>12)",
    ]
    cbar.locator = ticker.FixedLocator(tick_positions)
    cbar.formatter = ticker.FixedFormatter(icing_labels)
    cbar.update_ticks()

    plt.title(
        f"Weather Research and Forecasting Model\n"
        f"Average Grid Spacing: {avg_dx_km}x{avg_dy_km}km\n"
        f"Road Icing Index (Gaussian Smoothing = 0.5)",
        loc="left",
        fontsize=13,
    )
    plt.title(f"Valid: {hour}:{minute}Z {year}-{month}-{day}", loc="right", fontsize=13)

    output_file_name = f"wrf_{domain}_roadicing_{year}{month}{day}_{hour}_{minute}.png"
    plt.savefig(
        os.path.join(image_folder, output_file_name), bbox_inches="tight", dpi=150
    )
    plt.close()

    ds.close()
    wrf_file.close()


###############################################################################
# Main Processing Loop Over WRF Out Files (parallel with ProcessPoolExecutor) #
###############################################################################
if __name__ == "__main__":
    wrf_file_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))

    # Build argument list: each file gets its own previous file (for 1-hr precip diff)
    args_list = []
    for idx, wrf_file_path in enumerate(wrf_file_paths):
        prev_wrf_file_path = wrf_file_paths[idx - 1] if idx > 0 else None
        args_list.append((idx, wrf_file_path, prev_wrf_file_path, domain, image_folder))

    with ProcessPoolExecutor(max_workers=4) as executor:
        for _ in executor.map(process_wrf_file, args_list):
            pass

    print("Road Icing Index plot generation complete.")

    ###############################################################################
    # Build an animated GIF (if multiple .png files are found)
    ###############################################################################
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
    png_files.sort()

    if len(png_files) > 1:
        print("Creating .gif file from sorted .png files")
        images = [
            Image.open(os.path.join(image_folder, filename)) for filename in png_files
        ]
        gif_file_out = f"wrf_{domain}_Surface_roadicing.gif"
        gif_path = os.path.join(animation_folder, gif_file_out)
        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=800, loop=0
        )
        print("GIF generation complete.")
    else:
        print("Not enough images to create a GIF.")
