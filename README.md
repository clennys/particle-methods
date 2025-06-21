# Particle Methods and Computational Physics Simulations

**Course:** Particle Methods  
**Institution:** Università della Svizzera italiana (USI)  

A comprehensive collection of particle-based simulation methods for studying complex physical systems, including fire dynamics, ecological interactions, statistical mechanics, and molecular dynamics. This repository contains the practical implementations and coursework for the Particle Methods course at USI.

## Overview

This repository contains implementations of various particle-based computational methods commonly used in physics, ecology, and materials science. Each simulation demonstrates different aspects of complex systems modeling, from emergent behavior in biological systems to fundamental physics in statistical mechanics. The projects are designed as both educational tools and research-quality implementations for the USI Particle Methods course.

## Project Structure

The repository is organized into several independent simulation packages:

```
particle-methods/
├── forest_fire_model/          # Wildfire spread simulation using particle-based fire dynamics
├── habitat/python/predator_prey/   # Ecological predator-prey dynamics simulation
├── ising_model/                # 2D Ising model for statistical mechanics
├── molecular-dynamics-dissipative/  # Dissipative Particle Dynamics (DPD) simulation
└── molecular-dynamics-lennard-jones/  # Classical molecular dynamics with Lennard-Jones potential
```

## Simulations

### 🔥 Forest Fire Model
**Path:** `forest_fire_model/`

A sophisticated particle-based wildfire simulation that models fire spread across diverse terrain types with realistic environmental factors.

**Features:**
- Particle-based fire propagation with realistic physics
- Multiple terrain types (forest, coastal, wildland-urban interface, river systems)
- Environmental modeling (wind patterns, moisture gradients, fuel heterogeneity)
- Real-time visualization with 2D fire spread and 3D terrain views
- Comprehensive data collection and analysis tools

**Key Applications:** Wildfire research, emergency planning, environmental modeling

---

### 🐺🐰 Predator-Prey Simulation
**Path:** `habitat/python/predator_prey/`

An agent-based implementation of classic predator-prey dynamics simulating wolves and rabbits in a 2D spatial environment.

**Features:**
- Agent-based modeling with spatial movement patterns
- Efficient neighbor detection using spatial indexing
- Population dynamics visualization
- Configurable parameters for reproduction, movement, and life cycles

**Key Applications:** Ecological modeling, population dynamics research, complex systems studies

---

### ⚡ Ising Model
**Path:** `ising_model/`

A 2D Ising model implementation for studying magnetic phase transitions and critical phenomena.

**Features:**
- Monte Carlo simulation using Metropolis algorithm
- Temperature-dependent phase behavior analysis
- Magnetization and energy measurements
- Animation capabilities for lattice evolution
- Publication-quality plotting with gnuplot integration

**Key Applications:** Statistical mechanics, phase transitions, critical phenomena, condensed matter physics

---

### 💧 Dissipative Particle Dynamics (DPD)
**Path:** `molecular-dynamics-dissipative/`

A mesoscopic simulation technique for studying complex fluid phenomena, bridging microscopic and macroscopic scales.

**Features:**
- 2D DPD simulation with multiple particle types
- Support for Couette flow and Poiseuille flow scenarios
- Polymer chain and ring molecule modeling
- Comprehensive visualization of flow fields and molecular structures
- Real-time analysis of temperature, velocity, and density profiles

**Key Applications:** Fluid dynamics, polymer physics, soft matter research, microfluidics

---

### ⚛️ Molecular Dynamics (Lennard-Jones)
**Path:** `molecular-dynamics-lennard-jones/`

Classical molecular dynamics simulation for studying fundamental thermodynamic properties and phase behavior.

**Features:**
- 2D Lennard-Jones particle simulation
- Support for both NVE (microcanonical) and NVT (canonical) ensembles
- Efficient neighbor search using cell lists
- Real-time visualization of particle motion and energy evolution
- Radial Distribution Function (RDF) analysis
- Temperature control via Berendsen thermostat

**Key Applications:** Statistical mechanics, thermodynamics, phase behavior, molecular physics

## Installation and Usage

Each simulation package has its own installation requirements and usage instructions. Please refer to the individual README files in each project directory for specific setup and execution details.

### General Requirements
Most simulations require:
- Python 3.9+ with numpy, matplotlib, scipy
- Some projects use additional dependencies (see individual READMEs)
- C++ compiler for the Ising model

### Quick Start Examples

```bash
# Forest Fire Simulation
cd forest_fire_model
python -m forest_fire_model.main --map_type coastal --frames 1500

# Predator-Prey Dynamics
cd habitat/python/predator_prey
python -m predator_prey.main

# Molecular Dynamics
cd molecular-dynamics-lennard-jones
python -m molecular_dynamics.main --N 400 --steps 2000

# DPD Simulation
cd molecular-dynamics-dissipative
python -m molecular_dynamics.main --scenario couette --steps 5000
```

## Scientific Background

### Particle-Based Methods
These simulations employ various particle-based computational techniques:

- **Agent-Based Modeling (ABM):** Individual entities with simple rules producing complex emergent behavior
- **Molecular Dynamics (MD):** Newton's equations of motion for interacting particles
- **Monte Carlo (MC):** Statistical sampling methods for equilibrium properties
- **Dissipative Particle Dynamics (DPD):** Mesoscale method combining MD and fluid dynamics

### Physical Phenomena Studied
- **Phase transitions and critical phenomena** (Ising model)
- **Thermodynamic equilibrium and transport properties** (MD simulations)
- **Ecological population dynamics** (predator-prey interactions)
- **Environmental and disaster modeling** (wildfire spread)
- **Fluid dynamics and soft matter physics** (DPD)
