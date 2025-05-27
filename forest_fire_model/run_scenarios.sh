#!/bin/bash

# Multi-scenario runner script
# Usage: ./script.sh <scenario> [--save]
# Scenarios: 1, 2, 3, 4, 5, all

# Function to display usage
show_usage() {
    echo "Usage: $0 <scenario> [--save]"
    echo "Scenarios:"
    echo "  1     - Run scenario 1"
    echo "  2     - Run scenario 2" 
    echo "  3     - Run scenario 3"
    echo "  4     - Run scenario 4"
    echo "  5     - Run scenario 5"
    echo "  all   - Run all scenarios"
    echo ""
    echo "Options:"
    echo "  --save    Add --save flag to commands"
    echo ""
    echo "Examples:"
    echo "  $0 1"
    echo "  $0 3 --save"
    echo "  $0 all --save"
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Parse arguments
SCENARIO=$1
SAVE_FLAG=""

# Check for --save flag
if [ "$2" = "--save" ]; then
    SAVE_FLAG="--save"
fi

# Function to run scenario 1
run_scenario_1() {
    echo "Running Scenario 1 - Drought Firestorm (Forest Map)..."
    echo "Command: python -m forest_fire_model.main --map_type forest --wind_strength 0 --base_moisture 0 --fuel_types 6 --spread_rate 0.08 --ignition_probability 0.15 --intensity_decay 0.99 --random_strength 0.12 --particle_generation_rate 0.18 --burnout_rate 0.005 --min_intensity 0.1 --max_particles 800 --frames 1500 --multi_ignition --output drought_firestorm.mp4 --terrain_smoothness 10 --dt 0.25 $SAVE_FLAG"
    python -m forest_fire_model.main --map_type forest --wind_strength 0 --base_moisture 0 --fuel_types 6 --spread_rate 0.08 --ignition_probability 0.15 --intensity_decay 0.99 --random_strength 0.12 --particle_generation_rate 0.18 --burnout_rate 0.005 --min_intensity 0.1 --max_particles 800 --frames 1500 --multi_ignition --output drought_firestorm.mp4 --terrain_smoothness 10 --dt 0.25 $SAVE_FLAG
}

# Function to run scenario 2
run_scenario_2() {
    echo "Running Scenario 2 - WUI Fire (Wildland-Urban Interface)..."
    echo "Command: python -m forest_fire_model.env_vis --map_type wui --wind_strength 0.6 --wind_direction 0.75 --base_moisture 0.1 --fuel_types 8 --spread_rate 0.06 --ignition_probability 0.12 --intensity_decay 0.98 --particle_generation_rate 0.15 --burnout_rate 0.01 --particle_lifetime 20 --frames 1500 --output wui.mp4 --multi_ignition --dt 0.25 $SAVE_FLAG"
    python -m forest_fire_model.main --map_type wui --wind_strength 0.6 --wind_direction 0.75 --base_moisture 0.1 --fuel_types 8 --spread_rate 0.06 --ignition_probability 0.12 --intensity_decay 0.98 --particle_generation_rate 0.15 --burnout_rate 0.01 --particle_lifetime 20 --frames 1500 --output wui.mp4 --multi_ignition --dt 0.25 $SAVE_FLAG
}

# Function to run scenario 3
run_scenario_3() {
    echo "Running Scenario 3 - Forest Fire Model (River Map)..."
    echo "Command: python -m forest_fire_model.main --map_type river --wind_strength 0.9 --wind_direction 0.5 --ignite_x 15 --ignite_y 50 --spread_rate 0.06 --ignition_probability 0.12 --intensity_decay 0.96 --random_strength 0.15 --particle_generation_rate 0.14 --burnout_rate 0.02 --min_intensity 0.05 --particle_lifetime 25 --max_particles 600 --frames 2000 --output wind_spot_fires_v2.mp4 --terrain_smoothness 10 --fuel_types 6 --dt 0.25 $SAVE_FLAG"
    python -m forest_fire_model.main --map_type river --wind_strength 0.9 --wind_direction 0.5 --ignite_x 15 --ignite_y 50 --spread_rate 0.06 --ignition_probability 0.12 --intensity_decay 0.96 --random_strength 0.15 --particle_generation_rate 0.14 --burnout_rate 0.02 --min_intensity 0.05 --particle_lifetime 25 --max_particles 600 --frames 2000 --output wind_spot_fires_v2.mp4 --terrain_smoothness 10 --fuel_types 6 --dt 0.25 $SAVE_FLAG
}

# Function to run scenario 4
run_scenario_4() {
    echo "Running Scenario 4 - Coastal Fire (Variable Wind)..."
    echo "Command: python -m forest_fire_model.main --map_type coastal --variable_wind --wind_strength 0.4 --wind_direction 3.14 --base_moisture 0.25 --fuel_types 6 --spread_rate 0.04 --ignition_probability 0.08 --intensity_decay 0.97 --particle_generation_rate 0.08 --random_strength 0.08 --particle_lifetime 18 --frames 1500 --multi_ignition --dt 0.25 $SAVE_FLAG"
    python -m forest_fire_model.main --map_type coastal --variable_wind --wind_strength 0.4 --wind_direction 3.14 --base_moisture 0.25 --fuel_types 6 --spread_rate 0.04 --ignition_probability 0.08 --intensity_decay 0.97 --particle_generation_rate 0.08 --random_strength 0.08 --particle_lifetime 18 --frames 1500 --multi_ignition --dt 0.25 $SAVE_FLAG
}
# Function to run scenario 5
run_scenario_5() {
    echo "Running Scenario 5 - Climate Megafire (Mixed Terrain)..."
    echo "Command: python -m forest_fire_model.env_vis --map_type mixed --wind_strength 0.6 --wind_direction 0.78 --base_moisture 0.05 --fuel_types 6 --spread_rate 0.07 --ignition_probability 0.13 --intensity_decay 0.985 --random_strength 0.11 --particle_generation_rate 0.16 --burnout_rate 0.008 --min_intensity 0.08 --particle_lifetime 22 --max_particles 700 --multi_ignition --frames 1500 --output climate_megafire.mp4 --terrain_smoothness 10 --dt 0.25 $SAVE_FLAG"
    python -m forest_fire_model.main --map_type mixed --wind_strength 0.6 --wind_direction 0.78 --base_moisture 0.05 --fuel_types 6 --spread_rate 0.07 --ignition_probability 0.13 --intensity_decay 0.985 --random_strength 0.11 --particle_generation_rate 0.16 --burnout_rate 0.008 --min_intensity 0.08 --particle_lifetime 22 --max_particles 700 --multi_ignition --frames 1500 --output climate_megafire.mp4 --terrain_smoothness 10 --dt 0.25 $SAVE_FLAG
}

# Function to run all scenarios
run_all_scenarios() {
    echo "Running all scenarios..."
    echo "=========================="
    run_scenario_1
    echo ""
    run_scenario_2
    echo ""
    run_scenario_3
    echo ""
    run_scenario_4
    echo ""
    run_scenario_5
    echo "=========================="
    echo "All scenarios completed!"
}

# Main case statement
case $SCENARIO in
    1)
        run_scenario_1
        ;;
    2)
        run_scenario_2
        ;;
    3)
        run_scenario_3
        ;;
    4)
        run_scenario_4
        ;;
    5)
        run_scenario_5
        ;;
    all)
        run_all_scenarios
        ;;
    *)
        echo "Error: Invalid scenario '$SCENARIO'"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo "Script completed successfully!"
