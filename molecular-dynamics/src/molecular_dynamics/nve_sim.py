#!/usr/bin/env python3
"""
NVE simulation script that runs a complete analysis
of energy conservation with different parameters
"""
import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
import time
import os
from visualization import Visualizer


def run_nve_simulation(N, L=30.0, dt=0.01, steps=1000, temp=0.5, rc=2.5):
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

    Returns:
    --------
    dict: Simulation results and statistics
    """
    print(f"Running NVE simulation with N={N}, dt={dt}, steps={steps}")

    # Initialize simulation
    sim = Simulation(N=N, L=L, dt=dt, rc=rc, initial_temp=temp, use_thermostat=False)

    # Track energy drift
    initial_energy = sim.total_energy
    vis_steps = 10

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

    # Reset statistics after warmup
    sim.reset_measurements()

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
    sim.calculate_rdf(bins=50)
    vis.plot_energies()
    vis.plot_rdf()

    # Keep visualization window open
    vis.show()

        # Calculate statistics
    energy_data = np.array(sim.total_energy_history)
    initial_energy = energy_data[0]
    mean_energy = np.mean(energy_data)
    std_energy = np.std(energy_data)
    rel_fluctuation = std_energy / abs(mean_energy)
    rel_drift = (energy_data[-1] - energy_data[0]) / abs(energy_data[0])

    # Print summary
    print(
        f"\nSimulation completed in {elapsed:.2f} seconds ({steps_per_sec:.1f} steps/sec)"
    )
    print(f"Energy conservation statistics:")
    print(f"  Initial energy:    {initial_energy:.6f}")
    print(f"  Final energy:      {energy_data[-1]:.6f}")
    print(f"  Mean energy:       {mean_energy:.6f}")
    print(f"  Energy std dev:    {std_energy:.6f}")
    print(f"  Relative fluct:    {rel_fluctuation:.6%}")
    print(f"  Relative drift:    {rel_drift:.6%}")

    # Calculate RDF
    # sim.calculate_rdf(bins=50)

    # Return results
    return {
        "time_history": sim.time_history,
        "total_energy": sim.total_energy_history,
        "kinetic_energy": sim.kinetic_energy_history,
        "potential_energy": sim.potential_energy_history,
        "temperature": sim.temperature_history,
        "rdf_r": sim.rdf_r,
        "rdf": sim.rdf,
        "stats": {
            "steps": steps,
            "dt": dt,
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


def plot_results(results, title=None, output_dir="results"):
    """Plot simulation results"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Default title
    if title is None:
        title = f"N={results['stats']['N']}, dt={results['stats']['dt']}"

    # Energy plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["time_history"], results["kinetic_energy"], "r-", label="Kinetic")
    plt.plot(
        results["time_history"], results["potential_energy"], "g-", label="Potential"
    )
    plt.plot(results["time_history"], results["total_energy"], "b-", label="Total")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title(f"NVE Energy vs Time - {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/energy_{title.replace(' ', '_')}.png", dpi=150)

    # Temperature plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["time_history"], results["temperature"], "r-")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.title(f"Temperature vs Time - {title}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/temperature_{title.replace(' ', '_')}.png", dpi=150)

    # RDF plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["rdf_r"], results["rdf"], "b-")
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title(f"Radial Distribution Function - {title}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/rdf_{title.replace(' ', '_')}.png", dpi=150)

    plt.close("all")


def compare_time_steps(N=100, steps=1000, time_steps=[0.001, 0.005, 0.01, 0.02]):
    """Compare energy conservation with different time steps"""
    results = {}

    for dt in time_steps:
        print(f"\n=== Testing dt={dt} ===\n")
        results[dt] = run_nve_simulation(N=N, dt=dt, steps=steps)

        # Plot individual results
        plot_results(results[dt], title=f"N={N}, dt={dt}")

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
    plt.savefig(f"results/dt_comparison_N{N}.png", dpi=150)

    # Print comparison table
    print("\n=== Energy Conservation Comparison ===\n")
    print(f"{'Time Step':<10} {'Rel. Fluct.':<15} {'Rel. Drift':<15} {'Steps/sec':<10}")
    print("-" * 50)

    for dt in time_steps:
        stats = results[dt]["stats"]
        print(
            f"{dt:<10} {stats['rel_fluctuation']:<15.6%} {stats['rel_drift']:<15.6%} {stats['steps_per_sec']:<10.1f}"
        )


def compare_system_sizes(dt=0.01, steps=1000, sizes=[100, 225, 400, 625, 900]):
    """Compare energy conservation with different system sizes"""
    results = {}

    for N in sizes:
        print(f"\n=== Testing N={N} ===\n")
        results[N] = run_nve_simulation(N=N, dt=dt, steps=steps)

        # Plot individual results
        plot_results(results[N], title=f"N={N}, dt={dt}")

    # Create comparison plots
    plt.figure(figsize=(12, 8))
    for N in sizes:
        plt.plot(
            results[N]["time_history"],
            results[N]["total_energy"] / N,  # Normalize by system size
            label=f"N={N}",
        )

    plt.xlabel("Time")
    plt.ylabel("Total Energy per Particle")
    plt.title(f"NVE Energy Conservation Comparison with dt={dt}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"results/size_comparison_dt{dt}.png", dpi=150)

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


if __name__ == "__main__":
    # Part a: Basic NVE simulation with different parameters
    print("\n=== Part (a): Testing NVE ensemble with different parameters ===\n")

    # Compare different time steps
    print("\n=== Comparing different time steps ===\n")
    compare_time_steps(N=100, steps=1000, time_steps=[0.001, 0.005, 0.01, 0.02])

    # Compare different system sizes
    print("\n=== Comparing different system sizes ===\n")
    compare_system_sizes(dt=0.01, steps=1000, sizes=[100, 400, 900])

    print("\nAll tests completed. Results saved to 'results' directory.")
