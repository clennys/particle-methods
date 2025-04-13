#!/usr/bin/env python3

import argparse
import numpy as np
import time
from numba_simulation import SimulationNumba as Simulation
from visualization import Visualizer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="2D Lennard-Jones Particle Simulation (NVE)"
    )
    parser.add_argument("--N", type=int, default=100, help="Number of particles")
    parser.add_argument("--L", type=float, default=30.0, help="Box size")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step")
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of simulation steps"
    )
    parser.add_argument("--temp", type=float, default=0.5, help="Initial temperature")
    parser.add_argument("--rc", type=float, default=2.5, help="Cutoff radius")
    parser.add_argument(
        "--vis_steps", type=int, default=10, help="Visualization update frequency"
    )
    parser.add_argument(
        "--rdf_bins", type=int, default=50, help="Number of bins for RDF"
    )
    parser.add_argument(
        "--thermostat",
        action="store_true",
        default=False,
        help="Enable Berendsen thermostat",
    )
    parser.add_argument(
        "--tau_factor",
        type=float,
        default=0.0025,
        help="dt/τ factor for Berendsen thermostat",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print(f"Initializing simulation with {args.N} particles...")
    sim = Simulation(
        N=args.N,
        L=args.L,
        dt=args.dt,
        rc=args.rc,
        initial_temp=args.temp,
        use_thermostat=args.thermostat,
        tau_factor=args.tau_factor,
    )

    # Setup visualization
    vis = Visualizer(sim, update_interval=args.vis_steps)
    vis.setup()

    # Warmup phase
    print("Starting warmup...")
    warmup_steps = min(100, args.steps // 10)
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
    print(f"Starting main simulation for {args.steps} steps...")
    start_time = time.time()

    for step in range(args.steps):
        sim.step()

        # Update visualization periodically
        if step % args.vis_steps == 0:
            vis.update()


        # Print status periodically
        if step % 100 == 0:
            momentum = np.linalg.norm(sim.total_momentum)
            print(
                f"Step {step}/{args.steps}, T={sim.temperature:.3f}, "
                f"E_tot={sim.total_energy:.5f}, |P|={momentum:.6f}"
            )

    elapsed = time.time() - start_time
    steps_per_sec = args.steps / elapsed
    print(
        f"Simulation completed in {elapsed:.2f} seconds ({steps_per_sec:.1f} steps/sec)"
    )

    # Calculate and plot RDF at the end
    sim.calculate_rdf()
    vis.plot_energies()
    vis.plot_rdf()

    # Keep visualization window open
    vis.show()


if __name__ == "__main__":
    main()
