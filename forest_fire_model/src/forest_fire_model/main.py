from forest_fire_model.model import ForestFireModel
import numpy as np
from forest_fire_model.particles import CellState
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import argparse
import matplotlib.gridspec as gridspec
import os

def parse_arguments():
    """Parse command line arguments for the forest fire simulation."""
    parser = argparse.ArgumentParser(description='Forest Fire Particle Simulation')
    
    # Grid size parameters
    parser.add_argument('--width', type=int, default=100, help='Width of the simulation grid')
    parser.add_argument('--height', type=int, default=100, help='Height of the simulation grid')
    
    # Ignition parameters
    parser.add_argument('--ignite_x', type=int, default=20, help='X coordinate for initial ignition')
    parser.add_argument('--ignite_y', type=int, default=20, help='Y coordinate for initial ignition')
    
    # Wind parameters
    parser.add_argument('--wind_direction', type=float, default=np.pi/4, 
                        help='Wind direction in radians (0=east, pi/2=north)')
    parser.add_argument('--wind_strength', type=float, default=0.5, 
                        help='Wind strength (0-1)')
    
    # Environment parameters
    parser.add_argument('--fuel_types', type=int, default=3, 
                        help='Number of different fuel type patches')
    parser.add_argument('--base_moisture', type=float, default=0.2, 
                        help='Base moisture level (0-1)')
    
    # Simulation parameters
    parser.add_argument('--frames', type=int, default=200, 
                        help='Maximum number of simulation frames')
    parser.add_argument('--interval', type=int, default=100, 
                        help='Animation interval in milliseconds')
    parser.add_argument('--save', action='store_true', 
                        help='Save the animation to a file')
    parser.add_argument('--output', type=str, default='forest_fire_animation.mp4', 
                        help='Output filename for animation (if save is enabled)')
    
    return parser.parse_args()

def run_simulation():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create model with specified dimensions
    model = ForestFireModel(args.width, args.height)
    
    # Initialize environment
    model.initialize_random_terrain()
    model.set_uniform_wind(direction=args.wind_direction, strength=args.wind_strength)
    model.set_fuel_heterogeneity(fuel_types=args.fuel_types)
    model.set_moisture_gradient(base_moisture=args.base_moisture)
    
    # Set up some barriers (e.g., roads or rivers)
    for i in range(args.width // 3, args.width * 2 // 3):
        model.grid[i, args.height // 2] = CellState.EMPTY.value
    
    # Ignite fire at specified position
    model.ignite(args.ignite_x, args.ignite_y)
    
    # Data collection for statistics
    timestamps = [0]
    fuel_counts = [np.sum(model.grid == CellState.FUEL.value)]
    burning_counts = [np.sum(model.grid == CellState.BURNING.value)]
    burned_counts = [np.sum(model.grid == CellState.BURNED.value)]
    
    # Set up figure with two subplots
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    
    # Main simulation plot
    ax_sim = plt.subplot(gs[0])
    # Statistics plot
    ax_stats = plt.subplot(gs[1])
    
    # Create lines for the statistics plot
    line_fuel, = ax_stats.plot(timestamps, fuel_counts, 'g-', label='Unburned Fuel')
    line_burning, = ax_stats.plot(timestamps, burning_counts, 'r-', label='Burning')
    line_burned, = ax_stats.plot(timestamps, burned_counts, 'k-', label='Burned')
    
    ax_stats.set_xlabel('Time Step')
    ax_stats.set_ylabel('Cell Count')
    ax_stats.set_title('Fire Progression')
    ax_stats.legend()
    
    # Wind field visualization
    # We'll create a downsampled grid for the wind arrows to avoid overcrowding
    skip = 10  # Display every 10th wind vector
    x_grid = np.arange(0, args.width, skip)
    y_grid = np.arange(0, args.height, skip)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Initialize the wind quiver plot (will be updated in animation)
    quiver = None
    
    # Frame counter
    frame_count = 0
    
    # Run simulation with animation
    def animate(frame):
        nonlocal quiver, frame_count
        
        # Update the model
        active = model.update()
        frame_count += 1
        
        # Update statistics
        timestamps.append(frame_count)
        fuel_counts.append(np.sum(model.grid == CellState.FUEL.value))
        burning_counts.append(np.sum(model.grid == CellState.BURNING.value))
        burned_counts.append(np.sum(model.grid == CellState.BURNED.value))
        
        # Update the lines with new data
        line_fuel.set_data(timestamps, fuel_counts)
        line_burning.set_data(timestamps, burning_counts)
        line_burned.set_data(timestamps, burned_counts)
        
        # Adjust y-axis limits as needed
        ax_stats.relim()
        ax_stats.autoscale_view()
        
        # Clear simulation axis and redraw
        ax_sim.clear()
        model.visualize(ax_sim)
        
        # Update wind field visualization
        # Sample the wind field at grid points
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                if x < args.width and y < args.height:
                    wind_vec = model.wind_field[int(x), int(y)]
                    U[j, i] = wind_vec[0]  # Wind x component
                    V[j, i] = wind_vec[1]  # Wind y component
        
        # Remove previous quiver if it exists
        if quiver:
            quiver.remove()
        
        # Create new quiver plot
        quiver = ax_sim.quiver(X, Y, U, V, color='blue', scale=30, width=0.002)
        
        # Add a wind direction legend
        wind_strength = np.sqrt(U**2 + V**2).mean()
        ax_sim.text(5, 5, f"Wind: {wind_strength:.2f}", 
                 color='blue', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7))
        
        # Stop animation if fire is no longer active or max frames reached
        if not active or frame_count >= args.frames:
            print(f"Simulation ended at time step {frame_count}")
            print(f"Final state: {fuel_counts[-1]} unburned cells, {burned_counts[-1]} burned cells")
            anim.event_source.stop()
        
        return ax_sim, ax_stats
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=args.frames, interval=args.interval, blit=False)
    
    # Save animation if requested
    if args.save:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=15, metadata=dict(artist='ForestFireModel'), bitrate=1800)
        anim.save(args.output, writer=writer)
        print(f"Animation saved to {args.output}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()
