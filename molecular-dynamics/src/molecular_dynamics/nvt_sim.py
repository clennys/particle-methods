#!/usr/bin/env python3
"""
NVT simulation script that runs the simulations required for homework problem 3, part (b)
"""
import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
# from numba_simulation import SimulationNumba as Simulation
import time
import os
from visualization import Visualizer


def run_nvt_simulation(N, T, L=30.0, steps=2000, rc=2.5, tau_factor=0.0025):
    """
    Run an NVT simulation with Berendsen thermostat.

    Parameters:
    -----------
    N : int
        Number of particles
    T : float
        Target temperature
    L : float
        Box size
    steps : int
        Number of simulation steps
    rc : float
        Cutoff radius
    tau_factor : float
        dt/τ factor for Berendsen thermostat

    Returns:
    --------
    dict: Simulation results
    """
    print(f"Running NVT simulation with N={N}, T={T}, steps={steps}")

    # Initialize simulation with thermostat
    sim = Simulation(
        N=N, 
        L=L, 
        dt=0.001,  # Reasonable timestep based on part (a)
        rc=rc, 
        initial_temp=T, 
        use_thermostat=True,
        tau_factor=tau_factor
    )

    # Setup visualization
    vis = Visualizer(sim, update_interval=10)
    vis.setup()

    # Warmup phase to reach target temperature
    print("Starting warmup...")
    w_steps = 1500 if N == 900 else 1000
    # w_steps = 1000
    warmup_steps = max(w_steps, steps // 5)
    for step in range(warmup_steps):
        sim.step()
        if step % 20 == 0:
            print(
                f"Warmup step {step}/{warmup_steps}, T={sim.temperature:.3f}, "
                f"target T={T}"
            )
            vis.update()

    # Reset statistics after warmup
    # sim.reset_measurements()

    # Main simulation loop
    print(f"Starting main simulation for {steps} steps...")
    start_time = time.time()

    # We'll calculate the RDF periodically during equilibrium phase
    for step in range(steps):
        sim.step()

        # Update visualization periodically
        if step % 10 == 0:
            vis.update()

        # Print status periodically
        if step % 100 == 0:
            momentum = np.linalg.norm(sim.total_momentum)
            print(
                f"Step {step}/{steps}, T={sim.temperature:.3f} (target {T}), "
                f"E_tot={sim.total_energy:.5f}, |P|={momentum:.6f}"
            )

    elapsed = time.time() - start_time
    steps_per_sec = steps / elapsed
    print(
        f"Simulation completed in {elapsed:.2f} seconds ({steps_per_sec:.1f} steps/sec)"
    )

    # Calculate RDF with better statistics at the end
    sim.calculate_rdf(num_bins=100, max_samples=10)
    vis.plot_energies()
    vis.plot_rdf()

    # Keep visualization window open
    vis.show()

    # Return results
    return {
        "positions": sim.positions.copy(),
        "velocities": sim.velocities.copy(),
        "time_history": sim.time_history,
        "total_energy": sim.total_energy_history,
        "kinetic_energy": sim.kinetic_energy_history,
        "potential_energy": sim.potential_energy_history,
        "temperature": sim.temperature_history,
        "rdf_r": sim.rdf_r,
        "rdf": sim.rdf,
        "stats": {
            "N": N,
            "T": T,
            "steps": steps,
            "equilibrium_temp": np.mean(sim.temperature_history),
            "temp_std": np.std(sim.temperature_history),
            "elapsed_time": elapsed,
            "steps_per_sec": steps_per_sec,
        },
    }


def run_all_cases():
    """Run all the required NVT simulation cases"""
    # Create output directory for results
    os.makedirs("nvt_results", exist_ok=True)

    # Run all required cases
    cases = [
        {"N": 100, "T": 0.1},
        {"N": 100, "T": 1.0},
        {"N": 625, "T": 1.0},
        {"N": 900, "T": 1.0},
    ]

    results = {}
    
    for case in cases:
        N = case["N"]
        T = case["T"]
        
        print(f"\n{'='*50}")
        print(f"Running simulation with N={N}, T={T}")
        print(f"{'='*50}\n")
        
        case_results = run_nvt_simulation(N=N, T=T, steps=2000)
        case_key = f"N{N}_T{T}"
        results[case_key] = case_results
        
        # Save plots
        plt.figure(figsize=(10, 6))
        plt.plot(case_results["rdf_r"], case_results["rdf"], 'b-')
        plt.xlabel("r")
        plt.ylabel("g(r)")
        plt.title(f"Radial Distribution Function (N={N}, T={T})")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"nvt_results/rdf_N{N}_T{T}.png", dpi=150)
        
        plt.figure(figsize=(10, 6))
        plt.plot(case_results["time_history"], case_results["temperature"], 'r-')
        plt.axhline(y=T, color='k', linestyle='--', label=f"Target T={T}")
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.title(f"Temperature vs Time (N={N}, T={T})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"nvt_results/temp_N{N}_T{T}.png", dpi=150)
        
        # Close plots
        plt.close('all')
    
    # Print summary
    print("\n\nSimulation Results Summary:")
    print(f"{'Case':<15} {'Eq. Temp':<10} {'Temp Std':<10}")
    print("-" * 40)
    
    for case_key, case_results in results.items():
        stats = case_results["stats"]
        print(f"{case_key:<15} {stats['equilibrium_temp']:<10.4f} {stats['temp_std']:<10.4f}")

    print("\nAll simulations completed. Results saved to 'nvt_results' directory.")


if __name__ == "__main__":
    run_all_cases()
