#!/bin/bash

###############################################################################
# Wrapper script to:
#  - Activate the WRF Conda environment
#  - Locate the WRF /run directory
#  - Create a parent folder based on current UTC date
#  - Run a collection of Python post-processing scripts
#    for multiple domains and multiple point locations in Central Texas.
###############################################################################

# Initialize Conda environment
CONDA_BASE=$(conda info --base) || { echo "Failed to retrieve Conda base directory."; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh" || { echo "Failed to source Conda profile."; exit 1; }

# Activate the specific Conda environment
conda activate wrf-python || { echo "Failed to activate conda environment."; exit 1; }

# Define locations
declare -A locations=(
  ["KHYI"]="29.8927,-97.8630"  # San Marcos Regional Airport, San Marcos, Texas
  ["KAUS"]="30.1975,-97.6664"  # Austin-Bergstrom International Airport, Austin
  ["KSAT"]="29.5312,-98.4689"  # San Antonio International Airport, San Antonio
)

###############################################################################
# Helper function: run a list of gridded scripts in parallel for one domain
###############################################################################
run_scripts_in_parallel() {
  local domain="$1"
  shift
  local scripts=("$@")
  local counter=0
  local max_parallel=3   # Maximum number of scripts to run simultaneously

  mkdir -p "$parent_folder/$domain" || exit 1
  cd "$parent_folder/$domain" || exit 1

  for script in "${scripts[@]}"; do
    if [[ ! -f "$script_dir/$script" ]]; then
      echo "WARNING: missing script '$script_dir/$script' — skipping"
      continue
    fi

    echo "Running $script in domain $domain"
    python3 "$script_dir/$script" "$run_location" "$domain" &

    ((counter++))
    if [ "$counter" -eq "$max_parallel" ]; then
      wait
      counter=0
    fi
  done

  wait
  cd "$script_dir" || exit 1
}

###############################################################################
# Point-based scripts for each selected station
###############################################################################

# WBGT time series (DegF)
run_wbgt_timeseries() {
  local domain="$1"

  for location in "${!locations[@]}"; do
    local lat_long="${locations[$location]}"
    local lat long
    lat=$(echo "$lat_long" | cut -d',' -f1)
    long=$(echo "$lat_long" | cut -d',' -f2)

    mkdir -p "$parent_folder/$domain/$location" || { echo "Failed to create directory $parent_folder/$domain/$location"; exit 1; }
    cd "$parent_folder/$domain/$location" || { echo "Failed to cd into $parent_folder/$domain/$location"; exit 1; }

    echo "Running wbgt_solar_timeseries_degf.py for $location in $domain (lat=$lat lon=$long)"
    python3 "$script_dir/wbgt_solar_timeseries_degf.py" \
      "$run_location" "$domain" "$location" "$lat" "$long" 2>&1 || {
        echo "wbgt_solar_timeseries_degf.py failed for $location in $domain"
        exit 1
      }

    cd "$script_dir" || { echo "Failed to cd back to $script_dir"; exit 1; }
  done
}

# Vertical wind profile
run_vertical_wind() {
  local domain="$1"

  for location in "${!locations[@]}"; do
    local lat_long="${locations[$location]}"
    local lat long
    lat=$(echo "$lat_long" | cut -d',' -f1)
    long=$(echo "$lat_long" | cut -d',' -f2)

    mkdir -p "$parent_folder/$domain/$location" || { echo "Failed to create directory $parent_folder/$domain/$location"; exit 1; }
    cd "$parent_folder/$domain/$location" || { echo "Failed to cd into $parent_folder/$domain/$location"; exit 1; }

    echo "Running vertical_wind_profile.py for $location in $domain (lat=$lat lon=$long)"
    # Signature: <wrf_run_dir> <domain> <lat> <lon>
    python3 "$script_dir/vertical_wind_profile.py" \
      "$run_location" "$domain" "$lat" "$long" 2>&1 || {
        echo "vertical_wind_profile.py failed for $location in $domain"
        exit 1
      }

    cd "$script_dir" || { echo "Failed to cd back to $script_dir"; exit 1; }
  done
}

# Vertical wind profile 4km
run_vertical_wind_4km() {
  local domain="$1"

  for location in "${!locations[@]}"; do
    local lat_long="${locations[$location]}"
    local lat long
    lat=$(echo "$lat_long" | cut -d',' -f1)
    long=$(echo "$lat_long" | cut -d',' -f2)

    mkdir -p "$parent_folder/$domain/$location" || { echo "Failed to create directory $parent_folder/$domain/$location"; exit 1; }
    cd "$parent_folder/$domain/$location" || { echo "Failed to cd into $parent_folder/$domain/$location"; exit 1; }

    echo "Running vertical_wind_profile_4km.py for $location in $domain (lat=$lat lon=$long)"
    python3 "$script_dir/vertical_wind_profile_4km.py" \
      "$run_location" "$domain" "$lat" "$long" 2>&1 || {
        echo "vertical_wind_profile_4km.py failed for $location in $domain"
        exit 1
      }

    cd "$script_dir" || { echo "Failed to cd back to $script_dir"; exit 1; }
  done
}

# SkewT
run_skew_t() {
  local domain="$1"

  for location in "${!locations[@]}"; do
    local lat_long="${locations[$location]}"
    local lat long
    lat=$(echo "$lat_long" | cut -d',' -f1)
    long=$(echo "$lat_long" | cut -d',' -f2)

    mkdir -p "$parent_folder/$domain/$location" || { echo "Failed to create directory $parent_folder/$domain/$location"; exit 1; }
    cd "$parent_folder/$domain/$location" || { echo "Failed to cd into $parent_folder/$domain/$location"; exit 1; }

    echo "Running enhanced_skewt_diagram.py for $location in $domain (lat=$lat lon=$long)"
    python3 "$script_dir/enhanced_skewt_diagram.py" \
      "$run_location" "$domain" "$location" "$lat" "$long" 2>&1 || {
        echo "enhanced_skewt_diagram.py failed for $location in $domain"
        exit 1
      }

    cd "$script_dir" || { echo "Failed to cd back to $script_dir"; exit 1; }
  done
}

# Meteogram (DegF)
run_meteogram() {
  local domain="$1"

  for location in "${!locations[@]}"; do
    local lat_long="${locations[$location]}"
    local lat long
    lat=$(echo "$lat_long" | cut -d',' -f1)
    long=$(echo "$lat_long" | cut -d',' -f2)

    mkdir -p "$parent_folder/$domain/$location" || { echo "Failed to create directory $parent_folder/$domain/$location"; exit 1; }
    cd "$parent_folder/$domain/$location" || { echo "Failed to cd into $parent_folder/$domain/$location"; exit 1; }

    echo "Running meteogram_degf.py for $location in $domain (lat=$lat lon=$long)"
    python3 "$script_dir/meteogram_degf.py" \
      "$run_location" "$domain" "$location" "$lat" "$long" 2>&1 || {
        echo "meteogram_degf.py failed for $location in $domain"
        exit 1
      }

    cd "$script_dir" || { echo "Failed to cd back to $script_dir"; exit 1; }
  done
}

###############################################################################
# Find WRF /run directory
###############################################################################
find_wrf_run_directories() {
  local run_locations
  run_locations=$(find / -type d \
    \( -regex ".*/WRF[-Vv]*[0-9.]+/run" -o -regex ".*/WRF/run" \) 2>/dev/null)

  if [ -z "$run_locations" ]; then
    echo "No WRF run directory found. Exiting."
    exit 1
  else
    echo "WRF run folder found at the following location(s):"
    echo "$run_locations"
  fi

  if [ "$(echo "$run_locations" | wc -l)" -gt 1 ]; then
    echo "Multiple /run directories found:"
    echo "$run_locations"
    read -p "Enter the location of the /run directory you want to use: " run_location
  else
    run_location="$run_locations"
  fi
}

###############################################################################
# Main script body
###############################################################################
script_dir=$(pwd)
STARTYEAR=$(date +%Y --utc)
STARTMONTH=$(date +%m --utc)
STARTDAY=$(date +%d --utc)
parent_folder="${script_dir}/${STARTYEAR}_${STARTMONTH}_${STARTDAY}"
mkdir -p "$parent_folder"

find_wrf_run_directories

echo "Running point-based charts for domain d02."
run_wbgt_timeseries "d02"
sleep 5
run_meteogram "d02"
sleep 5
run_skew_t "d02"
sleep 5
run_vertical_wind "d02"
sleep 5
run_vertical_wind_4km "d02"

###############################################################################
# Scripts to run in parallel for domain d01
###############################################################################
d01_scripts=(
  "1000hpa_equiv_temp_k_pressure_wind_speed_dir.py"
  "250hpa_wind_height_isotachs.py"
  "300hpa_wind_height_isotachs.py"
  "500hpa_wind_height_isotachs.py"
  "700hpa_wind_height_isotachs.py"
  "850hpa_wind_height_isotachs.py"
  "850hpa_temp_advection_height_wind_speed_dir.py"
  "850hpa_frontogenesis.py"
  "850hpa_qvector_divergence_wind_pressure.py"
  "500hpa_vorticity_wind_pressure.py"
  "700hpa_relative_humidity_slp_thickness.py"
  "850hpa_temp_degc_height_wind_speed_dir.py"
  "925hpa_temp_degc_height_wind_speed_dir.py"
  "925hpa_wind_height_isotachs.py"
  "cloud_top_temperature.py"
  "precipitable_water_inch.py"
  "cloud_top_temperature_rainbow.py"
)

run_scripts_in_parallel "d01" "${d01_scripts[@]}"

###############################################################################
# Scripts to run in parallel for domain d02
###############################################################################
d02_scripts=(
  "convective_cape_cin.py"
  "cloud_frac_high_feet.py"
  "cloud_frac_low_feet.py"
  "cloud_frac_mid_feet.py"
  "cloud_top_temperature.py"
  "precipitable_water_inch.py"

  "surface_1hr_precip_inch_slp_isotherm.py"
  "surface_1hr_snow_inch_slp_isotherm.py"
  "surface_1hr_water_equivalent_snow_inch_slp_isotherm.py"

  "surface_24hr_precip_inch.py"
  "surface_24hr_snow_inch.py"
  "surface_24hr_water_equivalent_snow_inch.py"

  "surface_3hr_precip_inch.py"
  "surface_3hr_snow_inch.py"
  "surface_3hr_water_equivalent_snow_inch.py"

  "surface_dewpoint_degf_slp_wind_speed_dir.py"
  "surface_slp_wind_gust_speed_mph_direction.py"
  "surface_relative_humidity_slp_wind_speed_dir.py"
  "surface_simulated_radar_reflectivity.py"
  "surface_streamlines_terrain_ft.py"
  "surface_temp_degf_slp_wind_speed_dir.py"
  "surface_terrain_ft_slp_wind_speed_dir.py"

  "surface_daily_precip_inch.py"
  "surface_daily_snow_inch.py"
  "surface_daily_water_equivalent_snow_inch.py"
  "surface_total_precip_inch.py"
  "surface_total_snow_inch.py"
  "surface_total_water_equivalent_snow_depth_inch.py"

  "haines_index.py"
  "c_haines_index.py"
  "surface_heatindex_degf_slp_wind_speed_dir.py"
  "surface_humidex_degc_slp_wind_speed_dir.py"
  "surface_thi_degc_slp_wind_speed_dir.py"

  "surface_windchill_degf_slp_wind_speed_dir.py"
  "surface_visibility_miles.py"


  # Tropical (imperial-leaning: mph gust/speed + degF SST)
  "tropical_surface_slp_wind_speed_mph_direction.py"
  "tropical_surface_slp_wind_gust_speed_mph_direction.py"
  "tropical_surface_sst_degf_slp_wind_speed_dir.py"
)

run_scripts_in_parallel "d02" "${d02_scripts[@]}"

echo "All scripts have finished executing for all domains."
echo "Exiting conda environment."
conda deactivate

