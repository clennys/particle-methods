from forest_fire_model.model import ForestFireModel
from forest_fire_model.particles import CellState
import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

def parse_arguments():
    """Parse command line arguments for the combined environment and simulation visualization."""
    parser = argparse.ArgumentParser(description='Forest Fire Environment and Simulation Visualization')
    
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
    parser.add_argument('--variable_wind', action='store_true',
                        help='Use variable wind instead of uniform wind')
    
    # Environment parameters
    parser.add_argument('--fuel_types', type=int, default=3, 
                        help='Number of different fuel type patches')
    parser.add_argument('--base_moisture', type=float, default=0.2, 
                        help='Base moisture level (0-1)')
    parser.add_argument('--terrain_smoothness', type=int, default=5,
                        help='Terrain smoothness (higher = smoother)')
    
    # Simulation parameters
    parser.add_argument('--frames', type=int, default=200, 
                        help='Maximum number of simulation frames')
    parser.add_argument('--interval', type=int, default=100, 
                        help='Animation interval in milliseconds')
    
    # Output
    parser.add_argument('--save', action='store_true', 
                        help='Save simulation to a file')
    parser.add_argument('--output', type=str, default='forest_fire_animation.mp4',
                        help='Output filename for saved simulation')
    
    return parser.parse_args()

def run_combined_visualization():
    """Combined environment and fire simulation visualization."""
    args = parse_arguments()
    
    # Create model with specified dimensions
    model = ForestFireModel(args.width, args.height)
    
    # Initialize environment
    model.initialize_random_terrain(smoothness=args.terrain_smoothness)
    
    if args.variable_wind:
        model.set_variable_wind(base_direction=args.wind_direction, 
                               base_strength=args.wind_strength,
                               variability=0.2)
    else:
        model.set_uniform_wind(direction=args.wind_direction, 
                              strength=args.wind_strength)
    
    model.set_fuel_heterogeneity(fuel_types=args.fuel_types)
    model.set_moisture_gradient(base_moisture=args.base_moisture)
    
    # Set up some barriers (e.g., roads or rivers)
    for i in range(args.width // 3, args.width * 2 // 3):
        model.grid[i, args.height // 2] = CellState.EMPTY.value
    
    # Create figure with 3 subplots in a 2x2 grid
    # Top left: 3D terrain
    # Top right: Environment factors
    # Bottom: Simulation and stats
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
    
    # 3D terrain view
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Environment factors
    ax_env = fig.add_subplot(gs[0, 1])
    
    # Fire simulation
    ax_sim = fig.add_subplot(gs[1, 0])
    
    # Statistics
    ax_stats = fig.add_subplot(gs[1, 1])
    
    # Initialize the 3D plot and environmental factors plot
    
    # Create a meshgrid for 3D plotting
    x = np.arange(0, args.width, 1)
    y = np.arange(0, args.height, 1)
    X, Y = np.meshgrid(x, y)
    
    # Plot the terrain as a 3D surface
    terrain_surf = ax_3d.plot_surface(X, Y, model.terrain.T, 
                                   cmap='terrain', alpha=0.7,
                                   linewidth=0, antialiased=True)
    
    # Add colorbar for the terrain
    fig.colorbar(terrain_surf, ax=ax_3d, shrink=0.5, aspect=5, label='Elevation')
    
    # Set labels for 3D plot
    ax_3d.set_title("3D Terrain with Wind Vectors")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Elevation")
    
    # Initial setup for the environmental factors plot
    model.visualize_environment(ax=ax_env)
    
    # Initialize environment for fire simulation
    
    # Ignite fire
    model.ignite(args.ignite_x, args.ignite_y)
    
    # Data collection for statistics
    timestamps = [0]
    fuel_counts = [np.sum(model.grid == CellState.FUEL.value)]
    burning_counts = [np.sum(model.grid == CellState.BURNING.value)]
    burned_counts = [np.sum(model.grid == CellState.BURNED.value)]
    
    # Create lines for the statistics plot
    line_fuel, = ax_stats.plot(timestamps, fuel_counts, 'g-', label='Unburned Fuel')
    line_burning, = ax_stats.plot(timestamps, burning_counts, 'r-', label='Burning')
    line_burned, = ax_stats.plot(timestamps, burned_counts, 'k-', label='Burned')
    
    ax_stats.set_xlabel('Time Step')
    ax_stats.set_ylabel('Cell Count')
    ax_stats.set_title('Fire Progression')
    ax_stats.legend()
    
    # Initialize variables for animation
    fire_points_3d = None
    quiver_3d = None
    frame_count = 0
    
    # Run simulation with animation
    def animate(frame):
        nonlocal fire_points_3d, quiver_3d, frame_count
        
        # Update the model
        active = model.update()
        frame_count += 1
        
        # Update statistics
        timestamps.append(frame_count)
        fuel_counts.append(np.sum(model.grid == CellState.FUEL.value))
        burning_counts.append(np.sum(model.grid == CellState.BURNING.value))
        burned_counts.append(np.sum(model.grid == CellState.BURNED.value))
        
        # Update the statistics lines
        line_fuel.set_data(timestamps, fuel_counts)
        line_burning.set_data(timestamps, burning_counts)
        line_burned.set_data(timestamps, burned_counts)
        
        # Adjust y-axis limits as needed
        ax_stats.relim()
        ax_stats.autoscale_view()
        
        # Clear simulation axis and redraw
        ax_sim.clear()
        model.visualize(ax_sim)
        
        # Update 3D plot with fire particles
        # First remove old points if they exist
        if fire_points_3d:
            fire_points_3d.remove()
            
        # Get particle positions and intensities
        if model.particles:
            particle_x = [p.position[0] for p in model.particles]
            particle_y = [p.position[1] for p in model.particles]
            particle_z = [model.terrain[int(min(p.position[0], model.terrain.shape[0]-1)), 
                                       int(min(p.position[1], model.terrain.shape[1]-1))] + 0.1 
                         for p in model.particles]
            intensity = [p.intensity for p in model.particles]
            
            # Use intensity for point size and color
            sizes = [i * 30 for i in intensity]
            colors = [(1.0, min(1.0, i*0.5 + 0.5), 0.0) for i in intensity]  # Orange-yellow colors
            
            # Update fire particles on 3D plot
            fire_points_3d = ax_3d.scatter(particle_x, particle_y, particle_z, 
                                         c=colors, s=sizes, marker='o', alpha=0.7)
            
            # Update title with fire information
            ax_3d.set_title(f"3D Terrain - Time: {frame_count}, Active Fires: {len(model.particles)}")
            
            # Update the burned areas on the 3D plot by coloring the terrain
            if frame_count % 5 == 0:  # Update less frequently for performance
                # Get indices of burned cells
                burned_indices = np.where(model.grid == CellState.BURNED.value)
                if burned_indices[0].size > 0:
                    burned_x = burned_indices[0]
                    burned_y = burned_indices[1]
                    burned_z = [model.terrain[x, y] for x, y in zip(burned_x, burned_y)]
                    
                    # Add black dots for burned areas
                    ax_3d.scatter(burned_x, burned_y, burned_z, 
                                c='black', s=5, marker='s', alpha=0.3)
        
        # Stop animation if fire is no longer active or max frames reached
        if not active or frame_count >= args.frames:
            print(f"Simulation ended at time step {frame_count}")
            print(f"Final state: {fuel_counts[-1]} unburned cells, {burned_counts[-1]} burned cells")
            anim.event_source.stop()
        
        return ax_sim, ax_stats, ax_3d
    
    plt.tight_layout()
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=args.frames, interval=args.interval, blit=False)
    
    # Save animation if requested
    if args.save:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=15, metadata=dict(artist='ForestFireModel'), bitrate=1800)
        anim.save(args.output, writer=writer)
        print(f"Animation saved to {args.output}")
    
    plt.show()

if __name__ == "__main__":
    run_combined_visualization()
