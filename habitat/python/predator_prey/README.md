# Predator-Prey Simulation

A Python implementation of a classic predator-prey model simulating the interaction between wolves (predators) and rabbits (prey) in a two-dimensional space. This simulation demonstrates ecological dynamics including population cycles, spatial movement patterns, and emergent behavior.

## Features

- Agent-based modeling of predator-prey interactions
- Spatial movement with periodic boundary conditions
- Efficient neighbor detection using a grid-based spatial index
- Population dynamics visualization
- Configurable parameters (movement range, reproduction rates, etc.)
- Animation capabilities for real-time visualization

## Project Structure

```
predator_prey/
├── src/
│   └── predator_prey/
│       ├── __init__.py      # Package initialization
│       ├── agents.py        # Agent classes (Rabbit and Wolf)
│       ├── grid.py          # Spatial indexing grid
│       ├── sim.py           # Simulation engine
│       └── main.py          # Entry point with example parameters
├── pyproject.toml           # Project metadata and dependencies
├── poetry.lock              # Lock file for dependencies
└── README.md                # This file
```

### Key Components

- **agents.py**: Contains the base `Agent` class and its subclasses `Rabbit` and `Wolf`, implementing movement mechanics, reproduction, and life cycle.
- **grid.py**: Implements the spatial indexing system that efficiently tracks agent positions and enables fast neighbor lookups.
- **sim.py**: The core simulation engine that manages agent interactions, predation events, and population dynamics.
- **main.py**: Example script that demonstrates how to set up and run simulations with different parameters.

## Installation

### Using conda

1. Create a conda environment:
   ```bash
   conda create -n predator-prey python=3.13
   conda activate predator-prey
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Using Poetry (Alternative)

If you prefer using Poetry for dependency management:

```bash
poetry install
```

## Running the Simulation

### Basic Simulation

### What Happens When You Run the Default Script

When you run the main script without modifications, it will:

1. **Run Three Distinct Simulations**:
   - **Simulation 1**: The default configuration with a domain size of 10, 900 initial rabbits, 100 initial wolves, rabbit maximum age of 100, and standard movement.
   - **Simulation 2**: Same parameters but with reduced rabbit lifespan (max age 50), demonstrating how shorter prey lifespans affect the ecosystem.
   - **Simulation 3**: Reduced domain size (8) and significantly smaller movement steps (0.05), showing how limited movement affects predator-prey interactions.

2. **Monitor Progress**: During execution, the script will print status updates every 100 time steps, showing:
   ```
   Time step: 0, Rabbits: 900, Wolves: 100, Elapsed: 0.00s
   Time step: 100, Rabbits: X, Wolves: Y, Elapsed: Z.ZZs
   ...
   ```

3. **Generate Visualization Files**: It will create three PNG files showing population dynamics over time:
   - `population_dynamics_ex01.png`: Results from the first simulation
   - `population_dynamics_ex02.png`: Results from the second simulation 
   - `population_dynamics_ex03.png`: Results from the third simulation

4. **Display Population Plots**: Each simulation's population plot will be displayed, showing the classic predator-prey oscillations. These plots typically reveal:
   - Population cycles where predator and prey numbers fluctuate
   - Phase shifts between the two populations (predator peaks follow prey peaks)
   - The effect of different parameters on cycle amplitude and frequency

The entire process will take several minutes to complete as it runs each simulation for 4000 time steps.

```bash
# Activate your conda environment if not already active
conda activate predator-prey

# Run the main simulation script
python -m predator_prey.main
```

This will run several simulation examples with different parameters and save population dynamics plots as PNG files.

### Custom Simulation
```python
from predator_prey.sim import Simulation

# Create a simulation with custom parameters
sim = Simulation(
    domain_size=10,             # Size of the domain
    initial_rabbits=900,        # Initial rabbit population
    initial_wolves=100,         # Initial wolf population
    rabbit_max_age=100,         # Maximum age for rabbits
    wolf_hunger_threshold=50,   # Hunger threshold for wolves
    step_std=0.5,               # Standard deviation of movement step length
    capture_radius=0.5,         # Radius within which wolves can capture rabbits
    eat_prob=0.02,              # Probability of successful predation
    rabbit_repl_prob=0.02,      # Probability of rabbit reproduction
    n_subgrids=20               # Number of subgrids for spatial indexing
)

# Run simulation for 4000 time steps
sim.run(time_steps=4000)

# Plot and save population dynamics
sim.plot_population_dynamics("my_simulation")

# Optional: Run simulation with animation
# sim.run(time_steps=500, plot_interval=5)
```

## Simulation Parameters

- **domain_size**: Size of the square domain (with periodic boundary conditions)
- **initial_rabbits**: Initial number of rabbits
- **initial_wolves**: Initial number of wolves
- **rabbit_max_age**: Maximum lifespan of rabbits
- **wolf_hunger_threshold**: Number of time steps a wolf can survive without eating
- **step_std**: Standard deviation of the step length (controls movement distance)
- **capture_radius**: Radius within which wolves can capture rabbits
- **eat_prob**: Probability of a wolf successfully eating a rabbit within capture radius
- **rabbit_repl_prob**: Probability of rabbit reproduction per time step
- **n_subgrids**: Number of subgrids for spatial indexing (higher values improve performance for large populations)

## Visualization

The simulation provides two visualization options:

1. **Population Dynamics Plots**: Shows the population size of both species over time
   ```python
   sim.plot_population_dynamics("simulation_name")
   ```

2. **Real-time Animation**: Visualizes the spatial distribution and movement of agents
   ```python
   sim.run(time_steps=500, plot_interval=1)  # Update visualization every 5 steps
   ```
