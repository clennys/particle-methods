from forest_fire_model.model import ForestFireModel
from forest_fire_model.particles import CellState, FireParticle
from forest_fire_model.maps import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import time

def create_enhanced_fuel_types(model, args):
    """Create enhanced fuel types with distinct characteristics and colors.
    
    This function preserves existing fuel patterns and only enhances areas
    that are still at default fuel type (1.0).
    """
    
    # Don't reset existing fuel types - preserve map-specific patterns
    # model.fuel_types = np.ones((model.width, model.height))  # REMOVED - was causing overwrites
    
    # Define fuel type characteristics
    fuel_patches = [
        {'type': 2.0, 'color': 'brown', 'name': 'Dry Brush'},      # Very flammable
        {'type': 1.5, 'color': 'darkgreen', 'name': 'Dense Forest'}, # High flammability
        {'type': 1.0, 'color': 'forestgreen', 'name': 'Mixed Forest'}, # Normal
        {'type': 0.7, 'color': 'olive', 'name': 'Light Forest'},    # Medium
        {'type': 0.4, 'color': 'yellowgreen', 'name': 'Grassland'}, # Low flammability
    ]
    
    # Create large patches of each fuel type, but only in areas with default fuel (1.0)
    num_patches_per_type = 3
    
    for fuel_patch in fuel_patches:
        fuel_value = fuel_patch['type']
        
        # Skip if this is the default fuel type (already set)
        if fuel_value == 1.0:
            continue
            
        for _ in range(num_patches_per_type):
            # Random center for patch
            center_x = np.random.randint(10, args.width - 10)
            center_y = np.random.randint(10, args.height - 10)
            
            # Random patch size
            patch_radius = np.random.randint(8, 20)
            
            # Create irregular patch using noise
            for x in range(max(0, center_x - patch_radius), 
                          min(args.width, center_x + patch_radius)):
                for y in range(max(0, center_y - patch_radius), 
                              min(args.height, center_y + patch_radius)):
                    
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    
                    # Only set fuel type if it's fuel (not empty/houses) AND still at default (1.0)
                    if (model.grid[x, y] == CellState.FUEL.value and 
                        distance <= patch_radius and
                        abs(model.fuel_types[x, y] - 1.0) < 0.1):  # Only modify default fuel areas
                        
                        # Add some randomness to patch edges
                        edge_probability = max(0, 1 - (distance / patch_radius) + 
                                             np.random.uniform(-0.3, 0.3))
                        
                        if np.random.random() < edge_probability:
                            model.fuel_types[x, y] = fuel_value
    
    print("Enhanced fuel type distribution while preserving map-specific patterns")
    return fuel_patches

def run_combined_visualization(args):
    """Combined environment and fire simulation visualization."""
    
    # Create model with specified dimensions and balanced spread parameters
    model = ForestFireModel(args.width, args.height,
                          spread_rate=args.spread_rate,
                          random_strength=args.random_strength,
                          intensity_decay=args.intensity_decay,
                          min_intensity=args.min_intensity,
                          ignition_probability=args.ignition_probability,
                          particle_generation_rate=args.particle_generation_rate,
                          initial_particles=args.initial_particles,
                          burnout_rate=args.burnout_rate,
                          particle_lifetime=args.particle_lifetime)
    
    # Initialize environment
    model.initialize_random_terrain(smoothness=args.terrain_smoothness)
    
    # Set base moisture and fuel types BEFORE map creation
    model.set_moisture_gradient(base_moisture=args.base_moisture)
    model.fuel_types = np.ones((model.width, model.height))  # Initialize base fuel types
    
    # Set wind field before map creation (maps may modify if needed)
    if args.variable_wind:
        model.set_variable_wind(base_direction=args.wind_direction, 
                               base_strength=args.wind_strength,
                               variability=0.2)
    else:
        model.set_uniform_wind(direction=args.wind_direction, 
                              strength=args.wind_strength)
    
    # Create map layout based on selected type (will modify moisture/fuel as needed)
    if not args.remove_barriers:
        if args.map_type == 'houses':
            create_small_city(model, args)
        elif args.map_type == 'river':
            create_river_map(model, args)
        elif args.map_type == 'wui':
            create_urban_map(model, args)  # Now creates WUI map with proper moisture modification
        elif args.map_type == 'coastal':
            create_mountain_map(model, args)  # Now creates coastal map with proper fuel transitions
        elif args.map_type == 'mixed':
            create_mixed_map(model, args)
        elif args.map_type == 'forest':
            print("Created pure forest map (no barriers)")
        
        print(f"Map type: {args.map_type}")
    
    # Apply enhanced fuel types only for forest maps or when explicitly requested
    if args.map_type == 'forest' or (args.fuel_types > 5 and args.map_type not in ['coastal', 'wui']):
        fuel_patches = create_enhanced_fuel_types(model, args)
    else:
        # Create a simple fuel patches list for legend consistency
        fuel_patches = [
            {'type': 2.0, 'color': 'brown', 'name': 'Dry Brush'},
            {'type': 1.5, 'color': 'darkgreen', 'name': 'Dense Forest'},
            {'type': 1.0, 'color': 'forestgreen', 'name': 'Mixed Forest'},
            {'type': 0.7, 'color': 'olive', 'name': 'Light Forest'},
            {'type': 0.4, 'color': 'yellowgreen', 'name': 'Grassland'},
        ]
    
    # Create custom visualization function with fuel type colors
    def visualize_with_fuel_colors(model, ax=None, show_particles=True):
        """Enhanced visualization with fuel type colors."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create fuel type background
        fuel_display = np.zeros((model.width, model.height, 3))  # RGB array
        
        # Define colors for each fuel type
        fuel_color_map = {
            0.4: [0.6, 0.8, 0.2],    # Grassland - yellow-green
            0.7: [0.4, 0.6, 0.2],    # Light Forest - olive
            1.0: [0.1, 0.4, 0.1],    # Mixed Forest - forest green
            1.5: [0.0, 0.3, 0.0],    # Dense Forest - dark green
            2.0: [0.5, 0.3, 0.1],    # Dry Brush - brown
        }
        
        # Set fuel colors
        for i in range(model.width):
            for j in range(model.height):
                if model.grid[i, j] == CellState.FUEL.value:
                    fuel_value = model.fuel_types[i, j]
                    # Find closest fuel type
                    closest_fuel = min(fuel_color_map.keys(), 
                                     key=lambda x: abs(x - fuel_value))
                    fuel_display[i, j] = fuel_color_map[closest_fuel]
                elif model.grid[i, j] == CellState.BURNING.value:
                    fuel_display[i, j] = [1.0, 0.0, 0.0]  # Red
                elif model.grid[i, j] == CellState.BURNED.value:
                    fuel_display[i, j] = [0.0, 0.0, 0.0]  # Black
                elif model.grid[i, j] == CellState.EMPTY.value:
                    fuel_display[i, j] = [0.7, 0.7, 0.9]  # Light blue
        
        # Display the fuel map
        ax.imshow(fuel_display.transpose(1, 0, 2), origin='lower', 
                 extent=[0, model.width, 0, model.height])
        
        # Add particles only if requested
        if show_particles and model.particles:
            particle_x = [p.position[0] for p in model.particles]
            particle_y = [p.position[1] for p in model.particles]
            intensity = [p.intensity for p in model.particles]
            
            sizes = [i * 30 for i in intensity]
            ax.scatter(particle_x, particle_y, s=sizes, color='yellow', alpha=0.7)
        
        # Add terrain contours only for right plot
        if show_particles:
            terrain_contour = ax.contour(
                np.arange(model.width), 
                np.arange(model.height), 
                model.terrain.T, 
                levels=8, 
                colors='white', 
                alpha=0.3, 
                linewidths=0.8
            )
        
        # Add statistics
        burned_percent = np.sum(model.grid == CellState.BURNED.value) / (model.width * model.height) * 100
        burning_count = np.sum(model.grid == CellState.BURNING.value)
        fuel_left = np.sum(model.grid == CellState.FUEL.value)
        
        info_text = (
            f"Fire spread: {model.fire_spread_distance:.1f} units\n"
            f"Burned: {burned_percent:.1f}%\n"
            f"Active fires: {burning_count}\n"
            f"Fuel remaining: {fuel_left}"
        )
        
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        ax.set_title(f"Fire Simulation with Particles - {args.map_type.title()}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        return ax
    
    # Create figure with 2 subplots side by side
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # 3D terrain view (left)
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Fire simulation (right)
    ax_right = fig.add_subplot(gs[0, 1])
    
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
    ax_3d.set_title(f"3D Terrain - {args.map_type.title()} Map")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Elevation")
    
    # Create legend for fuel types and fire states (for right plot)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.5, 0.3, 0.1], label='Dry Brush (Very Flammable)'),
        Patch(facecolor=[0.0, 0.3, 0.0], label='Dense Forest (High Flammable)'),
        Patch(facecolor=[0.1, 0.4, 0.1], label='Mixed Forest (Normal)'),
        Patch(facecolor=[0.4, 0.6, 0.2], label='Light Forest (Medium)'),
        Patch(facecolor=[0.6, 0.8, 0.2], label='Grassland (Low Flammable)'),
        Patch(facecolor=[1.0, 0.0, 0.0], label='Burning'),
        Patch(facecolor=[0.0, 0.0, 0.0], label='Burned'),
        Patch(facecolor=[0.7, 0.7, 0.9], label='Empty (Buildings/Water/Rock)'),
    ]
    
    # Ignite fire
    if args.multi_ignition:
        # Ignite multiple points for better spread
        ignite_points = [
            (args.ignite_x, args.ignite_y),
            (args.ignite_x + 10, args.ignite_y),
            (args.ignite_x - 10, args.ignite_y),
            (args.ignite_x, args.ignite_y + 10),
            (args.ignite_x, args.ignite_y - 10)
        ]
        for x, y in ignite_points:
            if 0 <= x < args.width and 0 <= y < args.height:
                model.ignite(x, y)
    else:
        model.ignite(args.ignite_x, args.ignite_y)
    
    # Initialize variables for animation
    fire_points_3d = None
    frame_count = 0
    last_frame_time = time.time()
    fps_history = []
    
    # Function to limit number of particles for performance
    def limit_particles(model, max_particles):
        if len(model.particles) > max_particles:
            # Sort particles by intensity (keep the most intense ones)
            model.particles.sort(key=lambda p: p.intensity, reverse=True)
            model.particles = model.particles[:max_particles]
            return True
        return False
    
    # Run simulation with animation
    def animate(frame):
        nonlocal fire_points_3d, frame_count, last_frame_time, fps_history
        
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - last_frame_time
        fps = 1.0 / max(elapsed, 0.001)  # Avoid division by zero
        fps_history.append(fps)
        if len(fps_history) > 20:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        last_frame_time = current_time
        
        # Limit particles for performance
        particles_limited = limit_particles(model, args.max_particles)
        
        # Update the model with custom time step size
        active = model.update(dt=args.dt)
        frame_count += 1
        
        # Clear right axis and redraw with fuel types and particles
        ax_right.clear()
        visualize_with_fuel_colors(model, ax_right, show_particles=True)
        
        # Add legend to right plot
        ax_right.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), 
                       fontsize=8, framealpha=0.9)
        
        # Add FPS counter and parameters to right plot
        performance_text = (
            f"FPS: {avg_fps:.1f}\n"
            f"Particles: {len(model.particles)}\n"
            f"Map: {args.map_type.title()}\n"
            f"Spread Rate: {args.spread_rate}\n"
            f"Ignition Prob: {args.ignition_probability}\n"
            f"Burned: {np.sum(model.grid == CellState.BURNED.value)} cells"
        )
        ax_right.text(0.02, 0.98, performance_text, transform=ax_right.transAxes, fontsize=9,
                      verticalalignment='top', horizontalalignment='left',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Update 3D visualization periodically - show realistic moving particles
        if frame_count % args.skip_3d_update == 0:
            # Remove old fire points if they exist
            if fire_points_3d:
                fire_points_3d.remove()
                fire_points_3d = None
            
            # Update 3D plot with fire particles (realistic moving visualization)
            if model.particles:
                # Get subset of particles to display
                display_particles = model.particles[:min(len(model.particles), args.particle_display_limit)]
                
                # Get particle positions and intensities
                particle_x = [p.position[0] for p in display_particles]
                particle_y = [p.position[1] for p in display_particles]
                particle_z = [model.terrain[int(min(p.position[0], model.terrain.shape[0]-1)), 
                                           int(min(p.position[1], model.terrain.shape[1]-1))] + 0.1 
                             for p in display_particles]
                intensity = [p.intensity for p in display_particles]
                
                # Use intensity for point size and color
                sizes = [i * 30 for i in intensity]
                colors = [(1.0, min(1.0, i*0.5 + 0.5), 0.0) for i in intensity]
                
                # Create realistic fire particles visualization
                fire_points_3d = ax_3d.scatter(particle_x, particle_y, particle_z, 
                                             c=colors, s=sizes, marker='o', alpha=0.7)
            
            # Add burned areas as black dots (less frequently for performance)
            burned_indices = np.where(model.grid == CellState.BURNED.value)
            if burned_indices[0].size > 0 and frame_count % (args.skip_3d_update * 2) == 0:
                # Sample burned areas for performance
                sample_size = min(1000, burned_indices[0].size)
                if burned_indices[0].size > sample_size:
                    sample_indices = np.random.choice(burned_indices[0].size, sample_size, replace=False)
                    burned_x = burned_indices[0][sample_indices]
                    burned_y = burned_indices[1][sample_indices]
                else:
                    burned_x = burned_indices[0]
                    burned_y = burned_indices[1]
                
                burned_z = [model.terrain[x, y] + 0.02 for x, y in zip(burned_x, burned_y)]
                
                ax_3d.scatter(burned_x, burned_y, burned_z, 
                            c='black', s=8, marker='s', alpha=0.6, label='Burned')
            
            # Update title with fire information
            ax_3d.set_title(f"3D {args.map_type.title()} - Time: {frame_count}, Active Fires: {len(model.particles)}")
        
        # Stop animation if fire is no longer active or max frames reached
        if not active or frame_count >= args.frames:
            print(f"Simulation ended at time step {frame_count}")
            print(f"Final state: {np.sum(model.grid == CellState.FUEL.value)} unburned cells, {np.sum(model.grid == CellState.BURNED.value)} burned cells")
            anim.event_source.stop()
        
        return ax_3d, ax_right
    
    plt.tight_layout()
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=args.frames, interval=args.interval, blit=False)
    
    # Save animation if requested
    if args.save:
        try:
            from matplotlib.animation import FFMpegWriter
            print(f"Saving animation to {args.output}...")
            writer = FFMpegWriter(fps=15, metadata=dict(artist='ForestFireModel'), bitrate=1800)
            anim.save(args.output, writer=writer)
            print(f"Animation saved successfully to {args.output}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Trying alternative writer...")
            try:
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=10)
                gif_output = args.output.replace('.mp4', '.gif')
                anim.save(gif_output, writer=writer)
                print(f"Animation saved as GIF to {gif_output}")
            except Exception as e2:
                print(f"Error saving as GIF: {e2}")
                print("Animation will only be displayed, not saved.")
    
    plt.show()
