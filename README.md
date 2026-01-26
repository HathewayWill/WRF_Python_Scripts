Hello meteorologists and atmospheric scientists around the globe,

The day is finally here. After months, and honestly years, of development, I am officially finished building the full suite of Python charting scripts I set out to create for NSF NCAR’s Weather Research and Forecasting model, WRF.

Over the years I have been developing Python tools that highlight WRF’s capabilities for operational forecasting and real-world analysis. These scripts were built not only to showcase WRF, but also to be shared with the community for free, especially for meteorologists and atmospheric scientists who may not have much Python experience but still want access to high-resolution graphics, high automation and batch processing, and a wide range of meteorological diagnostics and products. Every script has been standardized across the collection, so the workflow, structure, and commenting style are consistent, making it easier to understand what each script is doing, adapt it to your region, and modify it for your own needs.

In the Google Drive link and on my GitHub page, you will find a folder called “Python Charts that work World.” Inside that folder you will find 93 Python scripts and 5 Bash shell scripts that run the full suite of graphics, including multicore execution options. The products span synoptic and mesoscale dynamics, severe weather diagnostics, and simulated remote sensing, all based directly on WRF output. The goal is to make a complete, automated, operational-ready post-processing package that takes raw WRF output and turns it into briefing-quality graphics with minimal user intervention.

**GitHub profile:** 
[https://github.com/HathewayWill/](https://github.com/HathewayWill/)
**WRF-MOSIT repo:** 
[https://github.com/HathewayWill/WRF-MOSIT](https://github.com/HathewayWill/WRF-MOSIT/WRF-Python)

---

## Compatibility and installation

These scripts are designed to work with the WRF-MOSIT Conda environment using wrf-python, but they can also be used outside of Conda if you install the required Python modules:

**numpy, matplotlib, metpy, wrf-python, netCDF4, pillow, scipy, cartopy, geopandas**

---

## Map projections and WRF output handling

Each Python script can work with any WRF supported map projection, including **Polar, Lat-Lon, Mercator, and Lambert**. Each script is also designed to handle multiple WRF output patterns, including a single wrfout file, a single file containing multiple times, or multiple files containing multiple times. The intent is that you should not have to reorganize your output files into a special format to make the scripts work, the scripts are meant to adapt to how WRF is commonly run in real workflows.

---

## Geographic features, boundaries, and city labeling included in the plots

A major focus of this project was making the maps operationally useful the moment they are generated, without requiring manual GIS setup, shapefile hunting, or post-editing. Across the gridded map scripts you will see a consistent approach to adding geographic context through Cartopy features, plus automated city labeling for quick situational awareness, so the output is immediately interpretable in a forecasting or briefing environment.

### Cartopy map features used throughout

Many scripts include common Cartopy features so the meteorological fields remain anchored to real geography. Depending on the domain and product, this can include:

* Coastlines, with resolution appropriate for the map scale
* International borders and state or province boundaries, when relevant
* Lakes, rivers, and other water bodies for orientation
* Land and ocean shading or masking for clean contrast between continents and water
* Gridlines with labeled latitude and longitude ticks, formatted consistently for readability
* Optional terrain or topographic context where it matters, such as streamline and terrain plots

### Standardized geographic styling

Geographic elements were standardized across the collection so line weights, label sizes, and background features remain consistent from plot to plot. The goal is readability, geographic context supports the meteorological fields without overpowering them, and the overall look and feel stays familiar whether you are viewing upper-air diagnostics, surface fields, cloud products, or precipitation maps.

### City and location labeling

Many scripts include city labels to make the output immediately useful for operations and briefings. Labels are intended to support quick orientation and readability in typical WRF domains, often including major cities, airports, and other commonly referenced points, with placement chosen to reduce overlap with filled contours and line contours whenever possible.

### Designed to work globally

Because the scripts support **Polar, Lat-Lon, Mercator, and Lambert** projections, the geographic overlays and labeling logic are built to work anywhere in the world. The intent is that you can run the same workflow whether your domain is over North America, Europe, the Middle East, the tropics, or any other region, and still get a map that is geographically meaningful and presentation-ready.

---

## Metric and Imperial support

You will notice that some scripts have similar names. That is intentional. I developed both Metric and Imperial versions, so the same diagnostic can be produced in the most common unit sets used around the world. In many cases this means paired scripts like **millimeters versus inches, kilometers versus miles, meters versus feet, Celsius versus Fahrenheit, knots versus miles per hour**. The goal was for a forecaster or researcher to run the exact same suite regardless of location, without having to rewrite unit conversions or maintain separate code bases.

---

# Script categories and included Python filenames

## 1) Upper-Air Dynamics and Synoptic Pattern Diagnostics

**Description:** Pressure-level maps designed for diagnosing jet structure, trough and ridge placement, forcing for ascent, moisture transport, and overall synoptic and mesoscale evolution. These include classic operational analysis fields adapted for WRF output, produced automatically. Many of these plots combine multiple fields in a single graphic, for example height with isotachs and wind, or humidity with thickness and sea-level pressure, so the output supports real forecasting decisions rather than showing a single field in isolation.

* 1000hpa_equiv_temp_k_pressure_wind_speed_dir.py
* 250hpa_wind_height_isotachs.py
* 300hpa_wind_height_isotachs.py
* 500hpa_vorticity_wind_pressure.py
* 500hpa_wind_height_isotachs.py
* 700hpa_relative_humidity_slp_thickness.py
* 700hpa_wind_height_isotachs.py
* 850hpa_frontogenesis.py
* 850hpa_qvector_divergence_wind_pressure.py
* 850hpa_temp_advection_height_wind_speed_dir.py
* 850hpa_temp_degc_height_wind_speed_dir.py
* 850hpa_wind_height_isotachs.py
* 925hpa_temp_degc_height_wind_speed_dir.py
* 925hpa_wind_height_isotachs.py

![wrf_d01_250hPa_WIND_Hgt_Isotachs](https://github.com/user-attachments/assets/9669eb3e-269f-456c-b7a7-0f8de5bef45c)


---

## 2) Moisture, Clouds, and Column Diagnostics

**Description:** Products focused on moisture availability and cloud field structure for aviation forecasting, convective initiation context, and general cloud interpretation. These scripts are designed to quickly identify deep moisture plumes, saturation patterns, cloud field coverage by layer, and cloud-top thermal structure.

* cloud_frac_high_feet.py
* cloud_frac_high_meters.py
* cloud_frac_low_feet.py
* cloud_frac_low_meters.py
* cloud_frac_mid_feet.py
* cloud_frac_mid_meters.py
* cloud_top_temperature.py
* cloud_top_temperature_rainbow.py
* precipitable_water_cm.py
* precipitable_water_inch.py

![wrf_d02_Cloud_Top_Temp](https://github.com/user-attachments/assets/92bd61bc-dbb9-468d-b55d-ff2ff279290a)


---

## 3) Convective Environment and Severe Weather Diagnostics

**Description:** Stability and thermodynamic diagnostics intended to summarize the convective environment and changes in instability, inhibition, and parcel buoyancy. These are the types of fields forecasters commonly use to assess convective initiation potential, severe weather environment quality, and evolving mesoscale instability gradients.

* convective_cape_cin.py
* mixed_layer_lifted_index.py, slow to run, very computationally heavy
* surface_based_lifted_index.py, slow to run, very computationally heavy

![wrf_d02_CAPE_CIN](https://github.com/user-attachments/assets/dae3a5dc-5ab1-4e71-a86a-a627d6fa61f8)


---

## 4) Surface Analysis, Thermodynamics, and Human-Impact Indices

**Description:** Surface and near-surface products designed for operational forecasting, situational awareness, and briefing graphics. Many include sea-level pressure and wind overlays while emphasizing a target surface parameter. These scripts cover classic surface analysis variables plus human-impact indices commonly used in operations, including heat and cold stress metrics.

* surface_dewpoint_degc_slp_wind_speed_dir.py
* surface_dewpoint_degf_slp_wind_speed_dir.py
* surface_heatindex_degc_slp_wind_speed_dir.py
* surface_heatindex_degf_slp_wind_speed_dir.py
* surface_humidex_degc_slp_wind_speed_dir.py
* surface_relative_humidity_slp_wind_speed_dir.py
* surface_slp_wind_gust_speed_knots_direction.py
* surface_slp_wind_gust_speed_mph_direction.py
* surface_streamlines_terrain_ft.py
* surface_streamlines_terrain_m.py
* surface_temp_degc_slp_wind_speed_dir.py
* surface_temp_degf_slp_wind_speed_dir.py
* surface_terrain_ft_slp_wind_speed_dir.py
* surface_terrain_m_slp_wind_speed_dir.py
* surface_thi_degc_slp_wind_speed_dir.py
* surface_visibility_km.py
* surface_visibility_miles.py
* surface_windchill_degc_slp_wind_speed_dir.py
* surface_windchill_degf_slp_wind_speed_dir.py
* road_icing_index.py

![wrf_d02_SLP_WIND_Gust](https://github.com/user-attachments/assets/7eb83abf-eb7d-40a1-87cf-d9d236eca3a9)

![wrf_d02_Surface_Visibility_miles_AFWA](https://github.com/user-attachments/assets/860f2a18-3994-47f5-b284-9be88c7f6192)


---

## 5) Precipitation, Snowfall, and Snow Water Equivalent Accumulations

**Description:** Time-accumulated precipitation and winter-weather products at operationally useful intervals. These cover liquid precipitation, snowfall, and snow water equivalent across hourly, multi-hour, daily, and total accumulation windows. These products are meant to support forecasting of rainfall rates, storm totals, winter storm impacts, and hydrologic context, with Metric and Imperial variants built in.

* surface_1hr_precip_inch_slp_isotherm.py
* surface_1hr_precip_mm_slp_isotherm.py
* surface_1hr_snow_inch_slp_isotherm.py
* surface_1hr_snow_mm_slp_isotherm.py
* surface_1hr_water_equivalent_snow_inch_slp_isotherm.py
* surface_1hr_water_equivalent_snow_mm_slp_isotherm.py
* surface_24hr_precip_inch.py
* surface_24hr_precip_mm.py
* surface_24hr_snow_inch.py
* surface_24hr_snow_mm.py
* surface_24hr_water_equivalent_snow_inch.py
* surface_24hr_water_equivalent_snow_mm.py
* surface_3hr_precip_inch.py
* surface_3hr_precip_mm.py
* surface_3hr_snow_inch.py
* surface_3hr_snow_mm.py
* surface_3hr_water_equivalent_snow_inch.py
* surface_3hr_water_equivalent_snow_mm.py
* surface_daily_precip_inch.py
* surface_daily_precip_mm.py
* surface_daily_snow_inch.py
* surface_daily_snow_mm.py
* surface_daily_water_equivalent_snow_inch.py
* surface_daily_water_equivalent_snow_mm.py
* surface_total_precip_inch.py
* surface_total_precip_mm.py
* surface_total_snow_inch.py
* surface_total_snow_mm.py
* surface_total_water_equivalent_snow_depth_inch.py
* surface_total_water_equivalent_snow_depth_mm.py

![wrf_d02_3-hour_Total_Precip_SLP_Isotherm](https://github.com/user-attachments/assets/12b235b7-34cf-468f-a71c-76f1cd8e4b55)


---

## 6) Simulated Remote Sensing and Radar-Style Products

**Description:** Simulated observational-style products intended to provide a radar-like view of convective structure and precipitation intensity derived from WRF fields. This is meant to help bridge the gap between model output and what forecasters are used to interpreting in real time.

* surface_simulated_radar_reflectivity.py

![wrf_d02_SFC_Simulated_dBZ](https://github.com/user-attachments/assets/9c4fc4de-8b5a-4708-a154-c44454de167b)


---

## 7) Fire Weather Indices and Stability-Based Fire Diagnostics

**Description:** Fire weather focused diagnostics designed to highlight atmospheric stability and dryness patterns commonly used in wildfire forecasting and fire behavior discussions. These provide quick environmental context for plume-dominated fire potential and stability regimes.

* haines_index.py
* c_haines_index.py

![wrf_d02_CHaines_Index](https://github.com/user-attachments/assets/cda9909b-ed92-42c9-954f-ec5a71cdd17f)


---

## 8) Point-Based Forecast Graphics and Vertical Profile, Cross-Section Diagnostics

**Description:** Site-specific and vertical diagnostic plots for stations or user-defined latitude and longitude points. These complement the gridded maps with time series, thermodynamic profiles, and vertical structure views. They are meant for forecasting at specific airports, cities, incident locations, research sites, and decision support points.

* meteogram_degc.py
* meteogram_degf.py
* skewt_diagram.py
* enhanced_skewt_diagram.py
* vertical_wind_profile.py
* vertical_wind_profile_4km.py
* vertical_wind_relative_humidity_cross_section_height_pressure.py
* vertical_wind_speed_cross_section_height_pressure.py

![wrf_d02_SkewT_LogP](https://github.com/user-attachments/assets/be71126f-333d-4ab0-8370-78feec19ea96)

<img width="5000" height="6000" alt="meteogram_Cape Canaveral, FL_imperial" src="https://github.com/user-attachments/assets/ede4775c-33fd-47b7-976d-6d4bd4b79d9b" />


---

## 9) Tropical Surface Diagnostics

**Description:** Tropical-focused surface products emphasizing sea-level pressure, winds, gusts, and sea surface temperature, with Metric and Imperial options. These are designed to support tropical cyclone monitoring, trade wind regime diagnosis, and tropical marine forecasting workflows.

* tropical_surface_slp_wind_gust_speed_knots_direction.py
* tropical_surface_slp_wind_gust_speed_mph_direction.py
* tropical_surface_slp_wind_speed_knots_direction.py
* tropical_surface_slp_wind_speed_mph_direction.py
* tropical_surface_sst_degc_slp_wind_speed_dir.py
* tropical_surface_sst_degf_slp_wind_speed_dir.py

---

## How you run them

The five Bash shell scripts are designed to automate execution. They can activate the appropriate environment, locate or accept a WRF output directory, create date-stamped output folders, and run the chart suite in a structured way across domains, including multicore parallel processing where configured. The intent is that you can generate an entire forecast graphics package in one run, organized by domain and location, without manually running dozens of individual commands.

---

## Why I built this

The purpose of this project is simple, make high-quality WRF post-processing graphics easier for the broader community to use, including those who have never written a Python script. If you are already comfortable in Python, you can also treat this as a base framework, the scripts are standardized so you can add additional diagnostics in a consistent way, and keep your own additions aligned with the rest of the suite.
