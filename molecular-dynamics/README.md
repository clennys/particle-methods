# Molecular Dynamics Simulation

A 2D Lennard-Jones particle simulation for studying molecular dynamics in NVE (microcanonical) and NVT (canonical) ensembles.

## Features

- 2D molecular dynamics simulation of particles interacting through Lennard-Jones potential
- Support for both NVE (constant energy) and NVT (constant temperature) ensembles
- Efficient neighbor list implementation using cell lists for performance
- Real-time visualization of:
  - Particle positions
  - Energy components (kinetic, potential, total)
  - Temperature evolution
  - Radial Distribution Function (RDF)
- Statistical analysis of simulation results (energy conservation, temperature control)
- Configurable parameters (particle count, time step, temperature, etc.)

## Installation

You can install this package using either Conda or Poetry.

### Using Conda

1. Create a new conda environment from the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate mol_dyn
```

### Using Poetry

1. Install dependencies using Poetry:
```bash
poetry install
```

2. Activate the poetry environment:
```bash
eval $(poetry env activate)
```

## Usage

The package includes several simulation scripts for different purposes:

### Main Simulation

Run the main simulation with default parameters:

```bash
python -m molecular_dynamics.main
```

With custom parameters:

```bash
python -m molecular_dynamics.main --N 400 --L 40.0 --dt 0.005 --steps 2000 --temp 0.5 --rc 2.5 --vis_steps 10
```

### NVE Simulation (Constant Energy)

For studying energy conservation with different time steps and system sizes:

```bash
python -m molecular_dynamics.nve_sim
```

### NVT Simulation (Constant Temperature)

For studying phase behavior at different temperatures:

```bash
python -m molecular_dynamics.nvt_sim
```

## Command Line Arguments

The main simulation script supports the following arguments:

- `--N`: Number of particles (default: 100)
- `--L`: Box size (default: 30.0)
- `--dt`: Time step (default: 0.001)
- `--steps`: Number of simulation steps (default: 1000)
- `--temp`: Initial temperature (default: 0.5)
- `--rc`: Cutoff radius for interactions (default: 2.5)
- `--vis_steps`: Visualization update frequency (default: 10)
- `--rdf_bins`: Number of bins for RDF calculation (default: 50)
- `--thermostat`: Enable Berendsen thermostat (flag)
- `--tau_factor`: dt/τ factor for Berendsen thermostat (default: 0.0025)

## Performance Notes

The simulation employs cell lists to efficiently calculate forces, which significantly improves performance for large systems. However, the visualization can become slow and laggy when simulating large numbers of particles (N > 400). This is due to the overhead of updating the matplotlib plots in real-time.

If you need to run simulations with large particle counts, consider:
- Reducing the visualization update frequency (increase `--vis_steps`)
- Disabling visualization entirely for production runs by modifying the code
- Using the specialized scripts that save plots instead of displaying them in real-time

## Theory

### Lennard-Jones Potential

This simulation uses the Lennard-Jones potential to model particle interactions:

$$U(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$$

Where:
- $\epsilon$ is the depth of the potential well
- $\sigma$ is the distance at which the potential is zero
- $r$ is the distance between particles

In the simulation, reduced units are used with ε = 1 and σ = 1.

### Velocity Verlet Algorithm

The simulation uses the velocity Verlet algorithm for time integration:

1. Update velocities by half a time step: $v(t + \Delta t/2) = v(t) + (F(t)/m)(\Delta t/2)$
2. Update positions for a full time step: $r(t + \Delta t) = r(t) + v(t + \Delta t/2)\Delta t$
3. Calculate forces at the new positions: $F(t + \Delta t)$
4. Complete velocity update: $v(t + \Delta t) = v(t + \Delta t/2) + (F(t + \Delta t)/m)(\Delta t/2)$

### Berendsen Thermostat

For NVT simulations, the Berendsen thermostat is used to maintain constant temperature by scaling velocities:

$$\lambda = \sqrt{1 + \frac{\Delta t}{\tau}\left(\frac{T_0}{T} - 1\right)}$$

Where:
- $\lambda$ is the scaling factor
- $T_0$ is the target temperature
- $T$ is the current temperature
- $\tau$ is the coupling parameter

## Project Structure

- `src/molecular_dynamics/`
  - `main.py`: Main entry point for simulation
  - `simulation.py`: Core simulation engine
  - `cell_list.py`: Efficient neighbor search implementation
  - `visualization.py`: Real-time visualization tools
  - `nve_sim.py`: NVE ensemble analysis script
  - `nvt_sim.py`: NVT ensemble analysis script
