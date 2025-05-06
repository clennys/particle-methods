# Dissipative Particle Dynamics Simulation

A 2D molecular dynamics simulation implementing Dissipative Particle Dynamics (DPD) for studying complex fluid phenomena, including Couette flow, Poiseuille flow, and behavior of polymer chains and ring molecules.

## Features

- 2D DPD simulation with support for multiple particle types
- Efficient neighbor search using cell lists algorithm
- Support for various flow scenarios:
  - Test case with pure fluid particles
  - Couette flow with chain molecules
  - Poiseuille flow with ring molecules
- Comprehensive visualization of:
  - Particle positions and types
  - Bonds between particles
  - Temperature evolution
  - Velocity profiles
  - Density profiles
  - Energy components
- Real-time visualization during simulation
- Detailed final state analysis with exportable plots

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

The package includes several simulation scenarios:

### Test Simulation

Run a simple test simulation with only fluid particles:

```bash
python -m molecular_dynamics.main --scenario test --steps 5000 --dt 0.01
```

### Couette Flow Simulation

Simulate Couette flow with chain molecules (polymer chains):

```bash
python -m molecular_dynamics.main --scenario couette --steps 5000 --dt 0.01
```

### Poiseuille Flow Simulation

Simulate Poiseuille flow with ring molecules:

```bash
python -m molecular_dynamics.main --scenario poiseuille --steps 5000 --dt 0.01
```

## Command Line Arguments

The main simulation script supports the following arguments:

- `--scenario`: Simulation scenario to run (choices: "test", "couette", "poiseuille", default: "test")
- `--dt`: Time step (default: 0.01)
- `--steps`: Number of simulation steps (default: 5000)
- `--vis_steps`: Visualization update frequency (default: 10)
- `--output`: Output directory for results (default: "dpd_results")
- `--no_vis`: Disable visualization (for batch runs)

## Theoretical Background

### Dissipative Particle Dynamics (DPD)

DPD is a mesoscopic simulation technique that bridges the gap between microscopic methods (like Molecular Dynamics) and macroscopic methods (like Computational Fluid Dynamics). It represents fluid parcels as soft particles that interact through three types of forces:

1. **Conservative Force** (F^C):
   - Repulsive force that prevents particle overlap
   - Given by: F^C = a_ij (1 - r/r_c) r̂ for r < r_c, where a_ij is the maximum repulsion between particles i and j

2. **Dissipative Force** (F^D):
   - Friction force that reduces relative velocity between particles
   - Given by: F^D = -γ w^D(r) (r̂ · v_ij) r̂, where γ is the friction coefficient

3. **Random Force** (F^R):
   - Thermal noise that maintains system temperature
   - Given by: F^R = σ w^R(r) ξ_ij r̂, where σ is the noise amplitude and ξ_ij is a random number

The weight functions w^D and w^R are related by w^D(r) = [w^R(r)]² to satisfy the fluctuation-dissipation theorem. The noise amplitude and friction coefficient are related by σ² = 2γk_BT to ensure proper thermal equilibrium.

### Integration Algorithm

The simulation uses the velocity Verlet algorithm for time integration:

1. Update velocities by half a time step: v(t + Δt/2) = v(t) + (F(t)/m)(Δt/2)
2. Update positions for a full time step: r(t + Δt) = r(t) + v(t + Δt/2)Δt
3. Calculate forces at the new positions: F(t + Δt)
4. Complete velocity update: v(t + Δt) = v(t + Δt/2) + (F(t + Δt)/m)(Δt/2)

## Project Structure

- `src/molecular_dynamics/`
  - `main.py`: Main entry point and scenario configuration
  - `simulation.py`: Core DPD simulation engine
  - `cell_list.py`: Efficient neighbor search implementation
  - `forces.py`: Implementation of DPD forces and bond forces
  - `visualization.py`: Real-time visualization tools

## Flow Scenarios

### Couette Flow

Simulates fluid between two parallel plates where one plate moves relative to the other, creating a shear flow. The simulation includes chain molecules (A-A-B-B-B-B-B) to study polymer behavior in shear flow.

### Poiseuille Flow

Simulates pressure-driven flow through a channel, implemented as a body force applied to fluid particles. The simulation includes ring molecules to study their behavior in Poiseuille flow.

## Output

Simulation results are saved to the specified output directory, including:
- Temperature evolution plots
- Velocity profiles
- Density profiles
- Final state visualization
