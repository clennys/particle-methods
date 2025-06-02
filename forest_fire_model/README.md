# Forest Fire Model

A sophisticated particle-based forest fire simulation framework that models wildfire spread across diverse terrain types with realistic environmental factors including wind patterns, terrain effects, moisture gradients, and fuel heterogeneity.

## Overview

This simulation framework provides a comprehensive tool for studying wildfire behavior across different landscapes. It features:

- **Particle-based fire propagation** with realistic physics
- **Multiple terrain types** (forest, coastal, wildland-urban interface, river systems, mixed landscapes)
- **Environmental modeling** including wind fields, moisture gradients, terrain effects, and fuel types
- **Real-time visualization** with 2D fire spread and 3D terrain views
- **Comprehensive data collection** and analysis tools
- **Scenario-based simulations** for different fire conditions

## Features

### Core Simulation Engine
- **Advanced fire particle system** with intensity-based spread mechanics
- **Environmental factor integration** (wind, terrain, moisture, fuel types)
- **Multiple ignition strategies** (single point, multi-point, fuel zone-based)
- **Realistic fire behavior** including burnout rates, intensity decay, and spread patterns

### Terrain & Environment Types
- **Forest Maps**: Dense vegetation with varying fuel types
- **Coastal Maps**: Moisture gradients from sea to inland with seasonal winds
- **Wildland-Urban Interface (WUI)**: Mixed development with defensible space
- **River Systems**: Waterways with riparian vegetation patterns
- **Mixed Landscapes**: Complex terrain with multiple features

### Visualization & Analysis
- **Real-time dual-panel visualization** (2D fire spread + 3D terrain)
- **Comprehensive data collection** with time-series tracking
- **Advanced analysis tools** with statistical summaries and correlations
- **Scientific plotting** with publication-ready visualizations
- **Animation export** to MP4/GIF formats

## Installation

### Option 1: Using Poetry (Recommended)

Poetry manages dependencies and virtual environments automatically.

1. **Install Poetry** (if not already installed):
```bash
# Using pip
pip install poetry

# Using conda
conda install poetry

# Or using the official installer (Linux/macOS)
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Install the project and dependencies:**
```bash
cd forest_fire_model
poetry install
```

3. **Activate the virtual environment:**
```bash
eval $(poetry env activate)
```

4. **Run simulations:**
```bash
# Inside activated environment
python -m forest_fire_model.main --map_type forest --frames 1000

# Or directly with poetry (no activation needed)
poetry run python -m forest_fire_model.main --map_type forest --frames 1000
```

### Option 2: Using Conda with Environment File

1. **Create an environment.yml file** (example):
```yaml
# environment.yml
name: forest-fire
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - numpy>=2.2.5
  - matplotlib>=3.10.1
  - scipy>=1.15.2
  - pandas>=2.2.3
  - seaborn>=0.13.2
  - pip
  - pip:
    - black>=25.1.0
    - tqdm>=4.67.1
```

2. **Create environment from file:**
```bash
conda env create -f environment.yml
conda activate forest-fire
```

3. **Install the package:**
```bash
cd forest_fire_model
pip install -e .
```

### Option 3: Using Conda (Manual)

1. **Create a new conda environment:**
```bash
conda create -n forest-fire python=3.12
conda activate forest-fire
```

2. **Install dependencies via conda and pip:**
```bash
# Core scientific packages via conda
conda install numpy matplotlib scipy pandas seaborn

# Additional packages via pip
pip install black tqdm
```

3. **Install the package in development mode:**
```bash
cd forest_fire_model
pip install -e .
```

## Quick Start

### Basic Forest Fire Simulation
```bash
python -m forest_fire_model.main --map_type forest --frames 1000
```

### Coastal Fire with Wind Effects
```bash
python -m forest_fire_model.main \
    --map_type coastal \
    --wind_strength 0.8 \
    --wind_direction 0.2 \
    --multi_ignition \
    --frames 1500 \
    --save \
    --output coastal_fire.mp4
```

### WUI Fire Simulation
```bash
python -m forest_fire_model.main \
    --map_type wui \
    --wind_strength 0.6 \
    --base_moisture 0.1 \
    --spread_rate 0.06 \
    --multi_ignition \
    --frames 1500
```

## Usage Guide

### Command Line Parameters

#### Grid and Environment
```bash
--width WIDTH                # Grid width (default: 100)
--height HEIGHT              # Grid height (default: 100)
--map_type TYPE              # Map type: forest, coastal, wui, river, mixed (default: houses)
--terrain_smoothness N       # Terrain smoothness factor (default: 5)
--fuel_types N               # Number of fuel type patches (default: 3)
--base_moisture FLOAT        # Base moisture level 0-1 (default: 0.2)
```

#### Fire Parameters
```bash
--spread_rate FLOAT          # Fire spread rate 0-1 (default: 0.01)
--ignition_probability FLOAT # Ignition probability 0-1 (default: 0.03)
--intensity_decay FLOAT      # Intensity decay rate 0.9-0.99 (default: 0.97)
--burnout_rate FLOAT         # Cell burnout rate (default: 0.03)
--min_intensity FLOAT        # Minimum particle intensity (default: 0.2)
--particle_lifetime INT      # Particle lifetime in frames (default: 15)
--max_particles INT          # Maximum particle limit (default: 300)
```

#### Wind and Weather
```bash
--wind_direction FLOAT       # Wind direction in radians (default: π/4)
--wind_strength FLOAT        # Wind strength 0-1 (default: 0.5)
--variable_wind             # Use variable wind patterns
```

#### Ignition
```bash
--ignite_x INT              # X coordinate for ignition (default: 20)
--ignite_y INT              # Y coordinate for ignition (default: 20)
--multi_ignition            # Use multiple ignition points
```

#### Simulation Control
```bash
--frames INT                # Maximum simulation frames (default: 500)
--interval INT              # Animation interval in ms (default: 40)
--dt FLOAT                  # Time step size (default: 0.1)
--save                      # Save animation to file
--output FILENAME           # Output filename (default: forest_fire_animation.mp4)
```

### Predefined Scenarios

Use the provided scenario runner for quick access to realistic fire scenarios:

```bash
# Make the script executable
chmod +x run_scenarios.sh

# Run individual scenarios
./run_scenarios.sh 1          # Drought Firestorm
./run_scenarios.sh 2          # WUI Fire
./run_scenarios.sh 3          # Wind-driven Fire
./run_scenarios.sh 4          # Coastal Fire
./run_scenarios.sh 5          # Climate Megafire

# Run all scenarios
./run_scenarios.sh all

# Save animations
./run_scenarios.sh 3 --save
```

#### Available Scenarios

1. **Drought Firestorm**: Extreme dry conditions with multiple ignition points
2. **WUI Fire**: Wildland-urban interface with wind effects
3. **Wind-Driven River Fire**: Strong wind pushing fire across river terrain
4. **Enhanced Coastal Fire**: Scientific analysis with variable winds
5. **Climate Megafire**: Large-scale fire under extreme conditions

## Data Analysis

### Automatic Data Collection

The simulation automatically collects comprehensive data including:
- **Time series**: Particle counts, burn rates, spread distances, intensities
- **Spatial data**: Grid states, fuel maps, terrain, wind fields
- **Summary statistics**: Total burned area, peak intensities, fire duration

### Analysis Tools

Analyze saved simulation data:

```bash
# Analyze a single simulation
python -m forest_fire_model.analyze_sim simulation_data/fire_simulation_coastal_20250602_143022.pkl

# Quick summary only
python -m forest_fire_model.analyze_sim data.pkl --quick

# Analyze all simulations in directory
python -m forest_fire_model.analyze_sim --directory simulation_data/

# Compare multiple simulations
python -m forest_fire_model.analyze_sim file1.pkl file2.pkl --compare
```

### Generated Analysis Outputs

The analysis tool creates comprehensive reports including:

1. **Fire Progression Analysis**: Cell state evolution and spread patterns
2. **Particle Dynamics**: Particle behavior and intensity evolution
3. **Spatial Analysis**: Initial vs final states with environmental factors
4. **Burn Rate Analysis**: Temporal fire behavior patterns
5. **Environmental Effects**: Wind, moisture, and fuel type impacts
6. **Summary Statistics**: Key metrics and parameter correlations
7. **Fire Spread Patterns**: Progression stages and velocity analysis
8. **Report Summary**: Publication-ready overview with key findings

## Map Types and Characteristics

### Forest Maps
- **Dense vegetation** with varying fuel types (grassland to dry brush)
- **Heterogeneous fuel patches** creating realistic fire behavior
- **Terrain-influenced spread** with slope effects
- **Ideal for**: Studying pure wildfire dynamics

### Coastal Maps
- **Moisture gradients** from ocean to inland
- **Variable wind patterns** simulating sea/land breezes  
- **Fuel transitions** from salt-tolerant to inland vegetation
- **Natural barriers**: Seasonal creeks and ridges
- **Ideal for**: Coastal fire behavior, wind effects

### Wildland-Urban Interface (WUI)
- **Scattered housing clusters** with defensible space
- **Fuel management zones** around structures
- **Evacuation routes** and fire breaks
- **Mixed vegetation** and development patterns
- **Ideal for**: Structure protection strategies

### River Systems
- **Winding waterways** as natural fire breaks
- **Riparian vegetation** with higher moisture
- **Meandering corridors** affecting fire spread
- **Ideal for**: Natural barrier effectiveness

### Mixed Landscapes
- **Multiple terrain features** (rivers, towns, ridges)
- **Complex interactions** between different map elements
- **Realistic heterogeneity** for comprehensive studies
- **Ideal for**: Complex scenario analysis

## Advanced Configuration

### Custom Fire Parameters

Create specialized fire behaviors by adjusting key parameters:

```bash
# Slow-spreading ground fire
python -m forest_fire_model.main \
    --spread_rate 0.005 \
    --intensity_decay 0.99 \
    --burnout_rate 0.001 \
    --particle_lifetime 30

# Fast crown fire
python -m forest_fire_model.main \
    --spread_rate 0.15 \
    --ignition_probability 0.25 \
    --intensity_decay 0.95 \
    --particle_generation_rate 0.3

# Wind-driven fire
python -m forest_fire_model.main \
    --wind_strength 0.9 \
    --variable_wind \
    --spread_rate 0.08 \
    --random_strength 0.05
```

### Simulation Data
- **Pickle files**: Complete simulation data (`simulation_data/*.pkl`)
- **Time series**: Frame-by-frame measurements
- **Spatial snapshots**: Grid states at regular intervals
- **Summary statistics**: Aggregated metrics

### Visualizations
- **Animation files**: MP4 or GIF format
- **Analysis plots**: PNG format in analysis directories
- **Summary reports**: Comprehensive visual analysis
- **Statistical tables**: CSV format for further analysis

### Directory Structure
```
project_root/
├── simulation_data/          # Saved simulation data
│   ├── fire_simulation_*.pkl
│   └── ...
├── analysis_output/          # Analysis results
│   ├── fire_progression.png
│   ├── particle_dynamics.png
│   ├── spatial_analysis.png
│   └── report_summary.png
└── animations/              # Saved animations
    ├── coastal_fire.mp4
    └── ...
```

## Troubleshooting

### Common Issues

**Animation not saving:**
- Install FFmpeg: `conda install ffmpeg`
- Or use GIF format (automatic fallback)

**Performance issues:**
- Reduce `--max_particles` and `--particle_display_limit`
- Increase `--skip_3d_update` and `--interval`
- Use smaller grid dimensions

**Memory errors:**
- Reduce `--frames` and grid size
- Increase `--dt` for faster progression
- Lower `--max_particles`

**Analysis errors:**
- Ensure pickle files are complete (simulation finished)
- Check file permissions in analysis directory
- Verify all dependencies are installed
