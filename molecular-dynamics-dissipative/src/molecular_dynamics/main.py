#!/usr/bin/env python3

import numpy as np
import argparse
import time
import os
import matplotlib.pyplot as plt
from simulation import DPDSimulation
from visualization import DPDVisualizer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="2D Dissipative Particle Dynamics (DPD) Simulation"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="test",
        choices=["test", "couette", "poiseuille"],
        help="Simulation scenario to run",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument(
        "--steps", type=int, default=5000, help="Number of simulation steps"
    )
    parser.add_argument(
        "--vis_steps", type=int, default=10, help="Visualization update frequency"
    )
    parser.add_argument(
        "--output", type=str, default="dpd_results", help="Output directory for results"
    )
    parser.add_argument(
        "--no_vis", action="store_true", help="Disable visualization (for batch runs)"
    )
    return parser.parse_args()


def setup_test_simulation():
    print("Setting up test simulation with only fluid particles")

    gamma = 4.5
    kBT = 1.0
    sigma = np.sqrt(2 * gamma * kBT)  # = 3.0 for gamma=4.5, kBT=1.0

    sim = DPDSimulation(
        L=15.0,  # Box size
        density=4.0,  # Density
        dt=0.01,  # Time step
        rc=1.0,  # Cutoff radius
        sigma=sigma,  # Random force coefficient (calculated to satisfy thermostat relation)
        gamma=gamma,  # Dissipative force coefficient
        kBT=kBT,  # Temperature
    )

    # Default a_ij = 25 for fluid-fluid interactions
    a_matrix = np.full((4, 4), 25.0)
    sim.a_matrix = a_matrix

    return sim


def setup_couette_flow():
    print("Setting up Couette flow simulation with chain molecules")

    # Conservative force coefficient matrix for different particle types
    # F: fluid, W: wall, A: type A, B: type B
    # a_ij matrix from homework 4 part (b)
    a_matrix = np.array(
        [
            [25, 200, 25, 300],  # F (row) interactions with F, W, A, B (columns)
            [200, 0, 200, 200],  # W interactions with F, W, A, B
            [25, 200, 50, 25],  # A interactions with F, W, A, B
            [300, 200, 25, 1],  # B interactions with F, W, A, B
        ]
    )

    sim = DPDSimulation(
        L=15.0,  # Box size
        density=4.0,  # Density
        dt=0.01,  # Time step
        rc=1.0,  # Cutoff radius
        sigma=1.0,  # Random force coefficient
        gamma=4.5,  # Dissipative force coefficient
        kBT=1.0,  # Temperature
        a_matrix=a_matrix,  # Conservative force coefficients
        K_S=100.0,  # Bond spring constant
        r_S=0.1,  # Equilibrium bond length
    )

    # Create chain molecules (A-A-B-B-B-B-B)
    sim.create_chain_molecules(num_chains=42)

    sim.create_walls(thickness=1.0, positions=["y=0", "y=L"])

    sim.set_wall_velocity([5.0, 0.0])  # Bottom wall moves in positive x-direction

    bottom_wall = np.where((sim.types == sim.WALL) & (sim.positions[:, 1] < sim.rc))[0]
    top_wall = np.where(
        (sim.types == sim.WALL) & (sim.positions[:, 1] > sim.L - sim.rc)
    )[0]

    sim.velocities[bottom_wall] = [5.0, 0.0]
    sim.velocities[top_wall] = [-5.0, 0.0]

    print(
        f"Created Couette flow setup with {len(bottom_wall)} bottom wall and {len(top_wall)} top wall particles"
    )

    return sim


def setup_poiseuille_flow():
    print("Setting up Poiseuille flow simulation with ring molecules")

    # Conservative force coefficient matrix for different particle types
    # F: fluid, W: wall, A: type A
    # a_ij matrix from homework 4 part (c)
    a_matrix = np.array(
        [
            [25, 200, 25],  # F (row) interactions with F, W, A (columns)
            [200, 0, 200],  # W interactions with F, W, A
            [25, 200, 50],  # A interactions with F, W, A
        ]
    )

    # Extend to 4x4 for compatibility with the simulation class
    # The fourth row/column (TYPE_B) won't be used in this scenario
    extended_a_matrix = np.zeros((4, 4))
    extended_a_matrix[:3, :3] = a_matrix

    sim = DPDSimulation(
        L=15.0,  # Box size
        density=4.0,  # Density
        dt=0.01,  # Time step
        rc=1.0,  # Cutoff radius
        sigma=1.0,  # Random force coefficient
        gamma=4.5,  # Dissipative force coefficient
        kBT=1.0,  # Temperature
        a_matrix=extended_a_matrix,  # Conservative force coefficients
        K_S=100.0,  # Bond spring constant
        r_S=0.3,  # Equilibrium bond length for ring molecules
        body_force=0.3,  # Body force to drive flow
    )

    sim.create_ring_molecules(num_rings=10, ring_size=9)

    sim.create_walls(thickness=1.0, positions=["y=0", "y=L"])

    bottom_wall = np.where((sim.types == sim.WALL) & (sim.positions[:, 1] < sim.rc))[0]
    top_wall = np.where(
        (sim.types == sim.WALL) & (sim.positions[:, 1] > sim.L - sim.rc)
    )[0]


    sim.set_wall_velocity([0.0, 0.0])
    sim.velocities[bottom_wall] = [0.0, 0.0]
    sim.velocities[top_wall] = [-0.0, 0.0]


    return sim


def run_simulation(sim, args):
    os.makedirs(args.output, exist_ok=True)

    if not args.no_vis:
        vis = DPDVisualizer(sim, update_interval=args.vis_steps)
        vis.setup()
    else:
        vis = None

    print(f"Running simulation for {args.steps} steps with dt={args.dt}")
    sim.dt = args.dt

    start_time = time.time()

    for step in range(args.steps):
        sim.step()

        if vis and step % args.vis_steps == 0:
            vis.update()

        if step % 100 == 0:
            print(
                f"Step {step}/{args.steps}, T={sim.temperature:.3f}, t={sim.time:.2f}"
            )

            # Calculate and print momentum (conservation check for test case)
            momentum = np.sum(sim.velocities, axis=0)
            print(f"  Total momentum: [{momentum[0]:.6f}, {momentum[1]:.6f}]")

            # Example modification in run_simulation
            non_wall_indices = np.where(sim.types != sim.WALL)[0]
            if len(non_wall_indices) > 0:
                momentum = np.sum(sim.velocities[non_wall_indices], axis=0)
                print(f"  Total momentum (non-wall): [{momentum[0]:.6f}, {momentum[1]:.6f}]")
            else:
                print("  Total momentum (non-wall): [0.000000, 0.000000]")

            if args.scenario == "couette" and step % 500 == 0:
                _, vx_profile = sim.get_velocity_profile(direction="y", component="x")
                print(
                    f"  Velocity profile (min/max): {np.min(vx_profile):.3f}/{np.max(vx_profile):.3f}"
                )

            if args.scenario == "poiseuille" and step % 500 == 0:
                _, density_a = sim.get_density_profile(
                    direction="y", particle_type=sim.TYPE_A
                )
                print(
                    f"  Type A density profile (min/max): {np.min(density_a):.3f}/{np.max(density_a):.3f}"
                )

    elapsed = time.time() - start_time
    steps_per_sec = args.steps / elapsed
    print(
        f"Simulation completed in {elapsed:.2f} seconds ({steps_per_sec:.1f} steps/sec)"
    )

    save_results(sim, args)

    if vis:
        vis.plot_final_results(args)
        vis.show()

    return sim


def save_results(sim, args):
    plots_dir = os.path.join(args.output, args.scenario)
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(sim.time_history, sim.temperature_history, "r-")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.title(f"Temperature vs Time ({args.scenario})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "temperature.png"), dpi=150)

    if args.scenario == "couette":
        y_bins, vx_profile = sim.get_velocity_profile(direction="y", component="x")

        plt.figure(figsize=(10, 6))
        plt.plot(vx_profile, y_bins, "b.-")
        plt.xlabel("Velocity (x-component)")
        plt.ylabel("Position (y-direction)")
        plt.title("Couette Flow Velocity Profile")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "velocity_profile.png"), dpi=150)

        np.savetxt(
            os.path.join(plots_dir, "velocity_profile.csv"),
            np.column_stack((y_bins, vx_profile)),
            delimiter=",",
            header="y,vx",
        )

    if args.scenario == "poiseuille":
        y_bins, density_f = sim.get_density_profile(
            direction="y", particle_type=sim.FLUID
        )
        y_bins, density_a = sim.get_density_profile(
            direction="y", particle_type=sim.TYPE_A
        )

        plt.figure(figsize=(10, 6))
        plt.plot(y_bins, density_f, "b.-", label="Fluid")
        plt.plot(y_bins, density_a, "r.-", label="Ring (Type A)")
        plt.xlabel("Position (y-direction)")
        plt.ylabel("Number Density")
        plt.title("Poiseuille Flow Density Profiles")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "density_profile.png"), dpi=150)

        np.savetxt(
            os.path.join(plots_dir, "density_profile.csv"),
            np.column_stack((y_bins, density_f, density_a)),
            delimiter=",",
            header="y,density_fluid,density_A",
        )

        y_bins, vx_profile = sim.get_velocity_profile(direction="y", component="x")

        plt.figure(figsize=(10, 6))
        plt.plot(vx_profile, y_bins, "g.-")
        plt.xlabel("Velocity (x-component)")
        plt.ylabel("Position (y-direction)")
        plt.title("Poiseuille Flow Velocity Profile")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "velocity_profile.png"), dpi=150)

    print(f"Results saved to {plots_dir}")


def main():
    args = parse_arguments()

    print(f"Starting DPD simulation for scenario: {args.scenario}")

    if args.scenario == "test":
        sim = setup_test_simulation()
    elif args.scenario == "couette":
        sim = setup_couette_flow()
    elif args.scenario == "poiseuille":
        sim = setup_poiseuille_flow()
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    sim = run_simulation(sim, args)

    print("Simulation completed")


if __name__ == "__main__":
    main()
