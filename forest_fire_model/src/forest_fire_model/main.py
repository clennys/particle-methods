from forest_fire_model.model import ForestFireModel
from forest_fire_model.particles import CellState, FireParticle
from forest_fire_model.vis import run_combined_visualization
from forest_fire_model.maps import *
import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import time


def parse_arguments():
    """Parse command line arguments for the combined environment and simulation visualization."""
    parser = argparse.ArgumentParser(
        description="Forest Fire Environment and Simulation Visualization"
    )

    # Grid size parameters
    parser.add_argument(
        "--width", type=int, default=100, help="Width of the simulation grid"
    )
    parser.add_argument(
        "--height", type=int, default=100, help="Height of the simulation grid"
    )

    # Ignition parameters
    parser.add_argument(
        "--ignite_x", type=int, default=20, help="X coordinate for initial ignition"
    )
    parser.add_argument(
        "--ignite_y", type=int, default=20, help="Y coordinate for initial ignition"
    )
    parser.add_argument(
        "--multi_ignition",
        action="store_true",
        help="Ignite multiple points for better spread",
    )

    # Map selection
    parser.add_argument(
        "--map_type",
        type=str,
        default="houses",
        choices=["houses", "forest", "river", "wui", "coastal", "mixed"],
        help="Type of map layout to generate",
    )

    # Wind parameters
    parser.add_argument(
        "--wind_direction",
        type=float,
        default=np.pi / 4,
        help="Wind direction in radians (0=east, pi/2=north)",
    )
    parser.add_argument(
        "--wind_strength", type=float, default=0.5, help="Wind strength (0-1)"
    )
    parser.add_argument(
        "--variable_wind",
        action="store_true",
        help="Use variable wind instead of uniform wind",
    )

    # Environment parameters
    parser.add_argument(
        "--fuel_types",
        type=int,
        default=3,
        help="Number of different fuel type patches",
    )
    parser.add_argument(
        "--base_moisture", type=float, default=0.2, help="Base moisture level (0-1)"
    )
    parser.add_argument(
        "--terrain_smoothness",
        type=int,
        default=5,
        help="Terrain smoothness (higher = smoother)",
    )

    # Performance parameters
    parser.add_argument(
        "--max_particles",
        type=int,
        default=300,
        help="Maximum number of particles allowed (limits computational load)",
    )
    parser.add_argument(
        "--skip_3d_update",
        type=int,
        default=3,
        help="Only update 3D plot every N frames (higher = better performance)",
    )
    parser.add_argument(
        "--particle_display_limit",
        type=int,
        default=200,
        help="Maximum number of particles to display (improves rendering speed)",
    )

    # Slow spread parameters with good balance
    parser.add_argument(
        "--spread_rate",
        type=float,
        default=0.01,
        help="Base fire spread rate (0-1, higher = faster spread)",
    )
    parser.add_argument(
        "--ignition_probability",
        type=float,
        default=0.03,
        help="Base probability of cell ignition (0-1)",
    )
    parser.add_argument(
        "--intensity_decay",
        type=float,
        default=0.97,
        help="Intensity decay rate (higher = slower decay, 0.9-0.99)",
    )
    parser.add_argument(
        "--particle_lifetime", type=int, default=15, help="Base lifetime for particles"
    )
    parser.add_argument(
        "--random_strength",
        type=float,
        default=0.03,
        help="Strength of random movement (0-0.2)",
    )
    parser.add_argument(
        "--initial_particles", type=int, default=15, help="Number of initial particles"
    )
    parser.add_argument(
        "--particle_generation_rate",
        type=float,
        default=0.03,
        help="Rate at which new particles are generated",
    )
    parser.add_argument(
        "--burnout_rate",
        type=float,
        default=0.03,
        help="Rate at which burning cells burnout",
    )
    parser.add_argument(
        "--min_intensity",
        type=float,
        default=0.2,
        help="Minimum intensity before a particle dies",
    )

    # Simulation parameters
    parser.add_argument(
        "--frames", type=int, default=500, help="Maximum number of simulation frames"
    )
    parser.add_argument(
        "--interval", type=int, default=40, help="Animation interval in milliseconds"
    )
    parser.add_argument(
        "--remove_barriers",
        action="store_true",
        help="Remove barriers to allow fire to spread freely",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step size for simulation (smaller = slower progression)",
    )

    # Output
    parser.add_argument("--save", action="store_true", help="Save simulation to a file")
    parser.add_argument(
        "--output",
        type=str,
        default="forest_fire_animation.mp4",
        help="Output filename for saved simulation",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    run_combined_visualization(args)


if __name__ == "__main__":
    main()
