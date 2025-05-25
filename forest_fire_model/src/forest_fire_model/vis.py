from forest_fire_model.model import ForestFireModel
from forest_fire_model.particles import CellState, FireParticle
from forest_fire_model.maps import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import time
import pickle
import os
from datetime import datetime

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

class SimulationDataCollector:
    """Collects and stores simulation data for analysis"""
    
    def __init__(self, args):
        self.args = vars(args)  # Store all command line arguments
        self.simulation_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'parameters': self.args,
                'map_type': args.map_type,
                'wind_direction': args.wind_direction,
                'wind_strength': args.wind_strength,
                'grid_size': (args.width, args.height)
            },
            'time_series': {
                'frame': [],
                'active_particles': [],
                'burning_cells': [],
                'burned_cells': [],
                'fuel_cells': [],
                'fire_spread_distance': [],
                'total_ignited_cells': [],
                'burn_rate': [],  # cells burned per time step
                'particle_intensity_avg': [],
                'particle_intensity_max': [],
                'wind_effect_strength': []
            },
            'spatial_data': {
                'initial_grid': None,
                'final_grid': None,
                'fuel_types': None,
                'moisture_map': None,
                'terrain': None,
                'wind_field': None,
                'ignition_points': [],
                'burned_area_progression': []  # Store snapshots of burned areas
            },
            'summary_stats': {}
        }
        
    def collect_frame_data(self, model, frame_count):
        """Collect data for current frame"""
        # Count different cell types
        burning_count = np.sum(model.grid == CellState.BURNING.value)
        burned_count = np.sum(model.grid == CellState.BURNED.value) 
        fuel_count = np.sum(model.grid == CellState.FUEL.value)
        
        # Calculate burn rate (change in burned cells)
        if len(self.simulation_data['time_series']['burned_cells']) > 0:
            prev_burned = self.simulation_data['time_series']['burned_cells'][-1]
            burn_rate = burned_count - prev_burned
        else:
            burn_rate = burned_count
            
        # Particle statistics
        if model.particles:
            particle_intensities = [p.intensity for p in model.particles]
            avg_intensity = np.mean(particle_intensities)
            max_intensity = np.max(particle_intensities)
        else:
            avg_intensity = 0
            max_intensity = 0
            
        # Wind effect (average wind strength)
        wind_strength = np.mean(np.sqrt(model.wind_field[:,:,0]**2 + model.wind_field[:,:,1]**2))
        
        # Store data
        ts = self.simulation_data['time_series']
        ts['frame'].append(frame_count)
        ts['active_particles'].append(len(model.particles))
        ts['burning_cells'].append(burning_count)
        ts['burned_cells'].append(burned_count)
        ts['fuel_cells'].append(fuel_count)
        ts['fire_spread_distance'].append(model.fire_spread_distance)
        ts['total_ignited_cells'].append(burning_count + burned_count)
        ts['burn_rate'].append(burn_rate)
        ts['particle_intensity_avg'].append(avg_intensity)
        ts['particle_intensity_max'].append(max_intensity)
        ts['wind_effect_strength'].append(wind_strength)
        
        # Store burned area snapshot every 10 frames for progression analysis
        if frame_count % 10 == 0:
            burned_mask = (model.grid == CellState.BURNED.value).astype(int)
            self.simulation_data['spatial_data']['burned_area_progression'].append({
                'frame': frame_count,
                'burned_area': burned_mask.copy()
            })
    
    def collect_initial_data(self, model):
        """Store initial simulation state"""
        self.simulation_data['spatial_data']['initial_grid'] = model.grid.copy()
        self.simulation_data['spatial_data']['fuel_types'] = model.fuel_types.copy()
        self.simulation_data['spatial_data']['moisture_map'] = model.moisture.copy()
        self.simulation_data['spatial_data']['terrain'] = model.terrain.copy()
        self.simulation_data['spatial_data']['wind_field'] = model.wind_field.copy()
        if model.ignition_point:
            self.simulation_data['spatial_data']['ignition_points'].append(model.ignition_point)
    
    def collect_final_data(self, model):
        """Store final simulation state and calculate summary statistics"""
        self.simulation_data['spatial_data']['final_grid'] = model.grid.copy()
        
        # Calculate summary statistics
        total_cells = model.width * model.height
        fuel_cells = np.sum(model.grid == CellState.FUEL.value)
        burned_cells = np.sum(model.grid == CellState.BURNED.value)
        empty_cells = np.sum(model.grid == CellState.EMPTY.value)
        
        ts = self.simulation_data['time_series']
        
        self.simulation_data['summary_stats'] = {
            'total_simulation_time': len(ts['frame']),
            'final_burned_percentage': (burned_cells / total_cells) * 100,
            'final_fuel_remaining': fuel_cells,
            'max_fire_spread_distance': model.fire_spread_distance,
            'max_active_particles': max(ts['active_particles']) if ts['active_particles'] else 0,
            'peak_burning_cells': max(ts['burning_cells']) if ts['burning_cells'] else 0,
            'average_burn_rate': np.mean(ts['burn_rate']) if ts['burn_rate'] else 0,
            'fire_duration': len([x for x in ts['active_particles'] if x > 0]),
            'total_area_burned': burned_cells,
            'empty_area_percentage': (empty_cells / total_cells) * 100,
            'fire_intensity_peak': max(ts['particle_intensity_max']) if ts['particle_intensity_max'] else 0,
            'fire_intensity_average': np.mean([x for x in ts['particle_intensity_avg'] if x > 0]) if ts['particle_intensity_avg'] else 0
        }
    
    def save_data(self, filename=None):
        """Save collected data to pickle file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            map_type = self.args['map_type']
            filename = f"fire_simulation_{map_type}_{timestamp}.pkl"
        
        # Create data directory if it doesn't exist
        os.makedirs('simulation_data', exist_ok=True)
        filepath = os.path.join('simulation_data', filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.simulation_data, f)
        
        print(f"Simulation data saved to: {filepath}")
        return filepath

def run_combined_visualization(args):
    """Combined environment and fire simulation visualization with data collection."""
    
    # Initialize data collector
    data_collector = SimulationDataCollector(args)
    
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
    
    # Collect initial data
    data_collector.collect_initial_data(model)
    
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
    # This will be handled in the animation function with split legends
    
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
                # Record additional ignition points
                data_collector.simulation_data['spatial_data']['ignition_points'].append((x, y))
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
        
        # Collect data for this frame
        data_collector.collect_frame_data(model, frame_count)
        
        # Clear right axis and redraw with fuel types and particles
        ax_right.clear()
        visualize_with_fuel_colors(model, ax_right, show_particles=True)
        
        # Combined legend in top left (fuel types + fire states)
        combined_legend_elements = [
            # Fuel Types
            Patch(facecolor=[0.5, 0.3, 0.1], label='Dry Brush'),
            Patch(facecolor=[0.0, 0.3, 0.0], label='Dense Forest'),
            Patch(facecolor=[0.1, 0.4, 0.1], label='Mixed Forest'),
            Patch(facecolor=[0.4, 0.6, 0.2], label='Light Forest'),
            Patch(facecolor=[0.6, 0.8, 0.2], label='Grassland'),
            # Fire States
            Patch(facecolor=[1.0, 0.0, 0.0], label='Burning'),
            Patch(facecolor=[0.0, 0.0, 0.0], label='Burned'),
            Patch(facecolor=[0.7, 0.7, 0.9], label='Buildings/Water'),
        ]
        
        # Add combined legend to top left
        ax_right.legend(handles=combined_legend_elements, loc='upper left', 
                       bbox_to_anchor=(0.02, 0.98), fontsize=8, framealpha=0.9,
                       title='Map Elements', title_fontsize=9, ncol=2)
        ax_right.get_legend().get_title().set_fontweight('bold')
        
        # Add FPS counter and parameters to top right
        performance_text = (
            f"FPS: {avg_fps:.1f}\n"
            f"Particles: {len(model.particles)}\n"
            f"Map: {args.map_type.title()}\n"
            f"Spread Rate: {args.spread_rate}\n"
            f"Ignition Prob: {args.ignition_probability}"
        )
        ax_right.text(0.98, 0.98, performance_text, transform=ax_right.transAxes, fontsize=9,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
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
            
            # Collect final data and save
            data_collector.collect_final_data(model)
            data_file = data_collector.save_data()
            print(f"Simulation data saved to: {data_file}")
            
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
