#!/usr/bin/env python3
"""
Modified NVE simulation script for Homework 3, exercise 1 that allows easy testing of different
parameter combinations without requiring command line arguments.
"""
import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
import time
import os
from visualization import Visualizer


def run_nve_simulation(N, L=30.0, dt=0.01, steps=1000, temp=0.5, rc=2.5, vis_steps=10, save_final_plots=True):
    """
    Run an NVE simulation and analyze energy conservation.

    Parameters:
    -----------
    N : int
        Number of particles
    L : float
        Box size
    dt : float
        Time step
    steps : int
        Number of simulation steps
    temp : float
        Initial temperature
    rc : float
        Cutoff radius
    vis_steps : int
        Visualization update frequency
    save_final_plots : bool
        Whether to save the final visualization plots

    Returns:
    --------
    dict: Simulation results and statistics
    """
    print(f"Running NVE simulation with N={N}, dt={dt}, steps={steps}, T_init={temp}")

    # Initialize simulation with NVE ensemble (no thermostat)
    sim = Simulation(
        N=N, 
        L=L, 
        dt=dt, 
        rc=rc, 
        initial_temp=temp, 
        use_thermostat=False
    )

    # Setup visualization
    vis = Visualizer(sim, update_interval=vis_steps)
    vis.setup()

    # Warmup phase
    print("Starting warmup...")
    warmup_steps = min(100, steps // 10)
    for step in range(warmup_steps):
        sim.step()
        if step % 10 == 0:
            print(
                f"Warmup step {step}/{warmup_steps}, T={sim.temperature:.3f}, "
                f"E_tot={sim.total_energy:.5f}"
            )
            vis.update()

    # Reset statistics after warmup
    # sim.reset_measurements()

    # Main simulation loop
    print(f"Starting main simulation for {steps} steps...")
    start_time = time.time()

    for step in range(steps):
        sim.step()

        # Update visualization periodically
        if step % vis_steps == 0:
            vis.update()

        # Print status periodically
        if step % 100 == 0:
            momentum = np.linalg.norm(sim.total_momentum)
            print(
                f"Step {step}/{steps}, T={sim.temperature:.3f}, "
                f"E_tot={sim.total_energy:.5f}, |P|={momentum:.6f}"
            )

    elapsed = time.time() - start_time
    steps_per_sec = steps / elapsed
    print(
        f"Simulation completed in {elapsed:.2f} seconds ({steps_per_sec:.1f} steps/sec)"
    )

    # Calculate and plot RDF at the end
    sim.calculate_rdf(num_bins=50, max_samples=5)
    vis.plot_energies()
    vis.plot_rdf()

    # Calculate statistics
    energy_data = np.array(sim.total_energy_history)
    initial_energy = energy_data[0]
    mean_energy = np.mean(energy_data)
    std_energy = np.std(energy_data)
    rel_fluctuation = std_energy / abs(mean_energy)
    rel_drift = (energy_data[-1] - energy_data[0]) / abs(energy_data[0])

    # Print summary
    print(f"\nEnergy conservation statistics:")
    print(f"  Initial energy:    {initial_energy:.6f}")
    print(f"  Final energy:      {energy_data[-1]:.6f}")
    print(f"  Mean energy:       {mean_energy:.6f}")
    print(f"  Energy std dev:    {std_energy:.6f}")
    print(f"  Relative fluct:    {rel_fluctuation:.6%}")
    print(f"  Relative drift:    {rel_drift:.6%}")

    # Keep visualization window open
    vis.show()
    
    # Save the final visualization plots if requested
    if save_final_plots:
        case_name = f"N{N}_dt{dt}_T{temp}_final"
        os.makedirs("nve_results", exist_ok=True)
        
        # Save final energy plot
        fig_energy = plt.figure(figsize=(10, 6))
        plt.plot(sim.time_history, sim.kinetic_energy_history, "r-", label="Kinetic")
        plt.plot(sim.time_history, sim.potential_energy_history, "g-", label="Potential")
        plt.plot(sim.time_history, sim.total_energy_history, "b-", label="Total")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.title(f"NVE Energy vs Time - N={N}, dt={dt}, T_init={temp}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"nve_results/final_energy_{case_name}.png", dpi=150)
        plt.close(fig_energy)
        
        # Save final temperature plot
        fig_temp = plt.figure(figsize=(10, 6))
        plt.plot(sim.time_history, sim.temperature_history, "r-")
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.title(f"Temperature vs Time - N={N}, dt={dt}, T_init={temp}")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"nve_results/final_temp_{case_name}.png", dpi=150)
        plt.close(fig_temp)
        
        # Save final RDF plot
        fig_rdf = plt.figure(figsize=(10, 6))
        plt.plot(sim.rdf_r, sim.rdf, "b-")
        plt.xlabel("r")
        plt.ylabel("g(r)")
        plt.title(f"Radial Distribution Function - N={N}, dt={dt}, T_init={temp}")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"nve_results/final_rdf_{case_name}.png", dpi=150)
        plt.close(fig_rdf)
        
        print(f"Final plots saved to nve_results directory with prefix 'final_'")

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
            "dt": dt,
            "steps": steps,
            "initial_energy": initial_energy,
            "final_energy": energy_data[-1],
            "mean_energy": mean_energy,
            "std_energy": std_energy,
            "rel_fluctuation": rel_fluctuation,
            "rel_drift": rel_drift,
            "elapsed_time": elapsed,
            "steps_per_sec": steps_per_sec,
        },
    }


def save_results(results, case_name):
    """
    Save simulation results to files.
    
    Parameters:
    -----------
    results : dict
        Simulation results dictionary
    case_name : str
        Name to use for saved files
    """
    # Create output directory if it doesn't exist
    os.makedirs("nve_results", exist_ok=True)
    
    # Save energy plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["time_history"], results["kinetic_energy"], "r-", label="Kinetic")
    plt.plot(results["time_history"], results["potential_energy"], "g-", label="Potential")
    plt.plot(results["time_history"], results["total_energy"], "b-", label="Total")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title(f"NVE Energy vs Time - {case_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"nve_results/energy_{case_name}.png", dpi=150)
    
    # Save temperature plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["time_history"], results["temperature"], "r-")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.title(f"Temperature vs Time - {case_name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"nve_results/temp_{case_name}.png", dpi=150)
    
    # Save RDF plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["rdf_r"], results["rdf"], "b-")
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title(f"Radial Distribution Function - {case_name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"nve_results/rdf_{case_name}.png", dpi=150)
    
    # Save relative energy error plot
    plt.figure(figsize=(10, 6))
    e_initial = results["total_energy"][0]
    rel_error = [(e - e_initial)/abs(e_initial) for e in results["total_energy"]]
    plt.plot(results["time_history"], rel_error, "b-")
    plt.xlabel("Time")
    plt.ylabel("Relative Energy Error")
    plt.title(f"NVE Relative Energy Error - {case_name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"nve_results/error_{case_name}.png", dpi=150)
    
    # Close plots
    plt.close('all')


def compare_time_steps():
    """Compare energy conservation with different time steps"""
    # Parameters for time step comparison
    N = 100             # Number of particles
    steps = 1000        # Number of simulation steps
    time_steps = [0.001, 0.005, 0.01, 0.02]  # Different time steps to test
    
    # Create output directory
    output_dir = "nve_results"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}

    for dt in time_steps:
        print(f"\n=== Testing dt={dt} ===\n")
        # Run simulation with visualization for the first time step only
        # to avoid multiple visualization windows
        if dt == time_steps[0]:
            results[dt] = run_nve_simulation(N=N, dt=dt, steps=steps)
        else:
            # Disable visualization by setting vis_steps to a very large number
            results[dt] = run_nve_simulation(N=N, dt=dt, steps=steps, vis_steps=steps+1)

        # Save individual results
        save_results(results[dt], f"N{N}_dt{dt}")

    # Create comparison plots
    plt.figure(figsize=(12, 8))
    for dt in time_steps:
        plt.plot(
            results[dt]["time_history"], results[dt]["total_energy"], label=f"dt={dt}"
        )

    plt.xlabel("Time")
    plt.ylabel("Total Energy")
    plt.title(f"NVE Energy Conservation Comparison for N={N}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/dt_comparison_N{N}.png", dpi=150)

    # Create relative energy error plot (more informative for comparing timesteps)
    plt.figure(figsize=(12, 8))
    for dt in time_steps:
        e_initial = results[dt]["total_energy"][0]
        rel_error = [(e - e_initial)/abs(e_initial) for e in results[dt]["total_energy"]]
        plt.plot(
            results[dt]["time_history"], 
            rel_error,
            label=f"dt={dt}"
        )

    plt.xlabel("Time")
    plt.ylabel("Relative Energy Error")
    plt.title(f"NVE Energy Error Comparison for N={N}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/dt_error_comparison_N{N}.png", dpi=150)

    # Print comparison table
    print("\n=== Energy Conservation Comparison ===\n")
    print(f"{'Time Step':<10} {'Rel. Fluct.':<15} {'Rel. Drift':<15} {'Steps/sec':<10}")
    print("-" * 50)

    for dt in time_steps:
        stats = results[dt]["stats"]
        print(
            f"{dt:<10} {stats['rel_fluctuation']:<15.6%} {stats['rel_drift']:<15.6%} {stats['steps_per_sec']:<10.1f}"
        )
    
    # Display the comparison plots
    plt.figure(figsize=(12, 8))
    for dt in time_steps:
        plt.plot(
            results[dt]["time_history"], results[dt]["total_energy"], label=f"dt={dt}"
        )

    plt.xlabel("Time")
    plt.ylabel("Total Energy")
    plt.title(f"NVE Energy Conservation Comparison for N={N}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def compare_system_sizes():
    """Compare energy conservation with different system sizes"""
    # Parameters for system size comparison
    dt = 0.01           # Time step
    steps = 1000        # Number of simulation steps
    sizes = [100, 225, 400, 625]  # Different system sizes to test
    
    # Create output directory
    output_dir = "nve_results"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}

    for N in sizes:
        print(f"\n=== Testing N={N} ===\n")
        # Run simulation with visualization for the first size only
        if N == sizes[0]:
            results[N] = run_nve_simulation(N=N, dt=dt, steps=steps)
        else:
            # Disable visualization by setting vis_steps to a very large number
            results[N] = run_nve_simulation(N=N, dt=dt, steps=steps, vis_steps=steps+1)

        # Save individual results
        save_results(results[N], f"N{N}_dt{dt}")

    # Create comparison plots - normalized by particle number
    plt.figure(figsize=(12, 8))
    for N in sizes:
        plt.plot(
            results[N]["time_history"],
            [e/N for e in results[N]["total_energy"]],  # Normalize by system size
            label=f"N={N}",
        )

    plt.xlabel("Time")
    plt.ylabel("Total Energy per Particle")
    plt.title(f"NVE Energy Conservation Comparison with dt={dt}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/size_comparison_dt{dt}.png", dpi=150)

    # Create relative energy error plot 
    plt.figure(figsize=(12, 8))
    for N in sizes:
        e_initial = results[N]["total_energy"][0]
        rel_error = [(e - e_initial)/abs(e_initial) for e in results[N]["total_energy"]]
        plt.plot(
            results[N]["time_history"], 
            rel_error,
            label=f"N={N}"
        )

    plt.xlabel("Time")
    plt.ylabel("Relative Energy Error")
    plt.title(f"NVE Energy Error Comparison for different system sizes (dt={dt})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/size_error_comparison_dt{dt}.png", dpi=150)

    # Print comparison table
    print("\n=== Energy Conservation Comparison ===\n")
    print(
        f"{'System Size':<12} {'Rel. Fluct.':<15} {'Rel. Drift':<15} {'Steps/sec':<10}"
    )
    print("-" * 52)

    for N in sizes:
        stats = results[N]["stats"]
        print(
            f"{N:<12} {stats['rel_fluctuation']:<15.6%} {stats['rel_drift']:<15.6%} {stats['steps_per_sec']:<10.1f}"
        )
    
    # Display the comparison plots
    plt.figure(figsize=(12, 8))
    for N in sizes:
        plt.plot(
            results[N]["time_history"],
            [e/N for e in results[N]["total_energy"]],  # Normalize by system size
            label=f"N={N}",
        )

    plt.xlabel("Time")
    plt.ylabel("Total Energy per Particle")
    plt.title(f"NVE Energy Conservation Comparison with dt={dt}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def run_custom_cases():
    """Run custom set of NVE simulations with various parameters"""
    # Create output directory for results
    os.makedirs("nve_results", exist_ok=True)

    # A sensible set of cases for exploring NVE ensemble behavior
    cases = [
        # Varying particle numbers (density effects)
        {"N": 100, "dt": 0.01, "steps": 2000, "temp": 0.5},  # Base case - low density
        {"N": 400, "dt": 0.01, "steps": 2000, "temp": 0.5},  # Medium density
        {"N": 900, "dt": 0.01, "steps": 2000, "temp": 0.5},  # High density
        
        # Varying time steps (integration accuracy)
        {"N": 400, "dt": 0.005, "steps": 2000, "temp": 0.5},  # Small time step - better energy conservation
        {"N": 400, "dt": 0.02, "steps": 2000, "temp": 0.5},   # Large time step - worse energy conservation
        
        # Varying initial temperatures (phase behavior)
        {"N": 400, "dt": 0.01, "steps": 2000, "temp": 0.1},   # Low temp - possible solid-like behavior
        {"N": 400, "dt": 0.01, "steps": 2000, "temp": 1.0},   # High temp - gas-like behavior
    ]

    results = {}
    
    for i, case in enumerate(cases):
        N = case["N"]
        dt = case["dt"]
        steps = case["steps"]
        temp = case["temp"]
        
        print(f"\n{'='*50}")
        print(f"Running simulation {i+1}/{len(cases)}: N={N}, dt={dt}, T_init={temp}")
        print(f"{'='*50}\n")
        
        # For first case, show visualization; for others, suppress it to avoid too many windows
        vis_steps = 10 if i == 0 else steps + 1
        
        case_results = run_nve_simulation(
            N=N, dt=dt, steps=steps, temp=temp, vis_steps=vis_steps, save_final_plots=True
        )
        
        case_key = f"N{N}_dt{dt}_T{temp}"
        results[case_key] = case_results
        
        # Save plots for this case
        save_results(case_results, case_key)
    
    # Print summary
    print("\n\nSimulation Results Summary:")
    print(f"{'Case':<20} {'Rel. Fluct.':<15} {'Rel. Drift':<15} {'Steps/sec':<10}")
    print("-" * 60)
    
    for case_key, case_results in results.items():
        stats = case_results["stats"]
        print(
            f"{case_key:<20} {stats['rel_fluctuation']:<15.6%} "
            f"{stats['rel_drift']:<15.6%} {stats['steps_per_sec']:<10.1f}"
        )

    print("\nAll simulations completed. Results saved to 'nve_results' directory.")


def run_single_simulation():
    """Run a single NVE simulation with custom parameters"""
    # Adjust these parameters as needed
    N = 100        # Number of particles
    dt = 0.01      # Time step
    steps = 1000   # Number of simulation steps
    temp = 0.5     # Initial temperature
    
    print(f"Running single NVE simulation with N={N}, dt={dt}, steps={steps}, T_init={temp}")
    run_nve_simulation(N=N, dt=dt, steps=steps, temp=temp)


if __name__ == "__main__":
    # Run the custom cases with sensible parameter combinations
    run_custom_cases()
