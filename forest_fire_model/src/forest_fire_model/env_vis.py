from forest_fire_model.model import ForestFireModel
from forest_fire_model.particles import CellState, FireParticle
import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import time

def parse_arguments():
    """Parse command line arguments for the combined environment and simulation visualization."""
    parser = argparse.ArgumentParser(description='Forest Fire Environment and Simulation Visualization')
    
    # Grid size parameters
    parser.add_argument('--width', type=int, default=100, help='Width of the simulation grid')
    parser.add_argument('--height', type=int, default=100, help='Height of the simulation grid')
    
    # Ignition parameters
    parser.add_argument('--ignite_x', type=int, default=20, help='X coordinate for initial ignition')
    parser.add_argument('--ignite_y', type=int, default=20, help='Y coordinate for initial ignition')
    parser.add_argument('--multi_ignition', action='store_true', 
                        help='Ignite multiple points for better spread')
    
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
    
    # Performance parameters
    parser.add_argument('--max_particles', type=int, default=300,
                        help='Maximum number of particles allowed (limits computational load)')
    parser.add_argument('--skip_3d_update', type=int, default=3,
                        help='Only update 3D plot every N frames (higher = better performance)')
    parser.add_argument('--particle_display_limit', type=int, default=200,
                        help='Maximum number of particles to display (improves rendering speed)')
    
    # Slow spread parameters with good balance
    parser.add_argument('--spread_rate', type=float, default=0.01,
                        help='Base fire spread rate (0-1, higher = faster spread)')
    parser.add_argument('--ignition_probability', type=float, default=0.03,
                        help='Base probability of cell ignition (0-1)')
    parser.add_argument('--intensity_decay', type=float, default=0.97,
                        help='Intensity decay rate (higher = slower decay, 0.9-0.99)')
    parser.add_argument('--particle_lifetime', type=int, default=15,
                        help='Base lifetime for particles')
    parser.add_argument('--random_strength', type=float, default=0.03,
                        help='Strength of random movement (0-0.2)')
    parser.add_argument('--initial_particles', type=int, default=15,
                        help='Number of initial particles')
    parser.add_argument('--particle_generation_rate', type=float, default=0.03,
                        help='Rate at which new particles are generated')
    parser.add_argument('--burnout_rate', type=float, default=0.03,
                        help='Rate at which burning cells burnout')
    parser.add_argument('--min_intensity', type=float, default=0.2,
                        help='Minimum intensity before a particle dies')
    
    # Simulation parameters
    parser.add_argument('--frames', type=int, default=500, 
                        help='Maximum number of simulation frames')
    parser.add_argument('--interval', type=int, default=40, 
                        help='Animation interval in milliseconds')
    parser.add_argument('--remove_barriers', action='store_true',
                        help='Remove barriers to allow fire to spread freely')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step size for simulation (smaller = slower progression)')
    
    # Auto-recovery to prevent fire from dying out
    parser.add_argument('--auto_recovery', action='store_true', default=True,
                        help='Enable automatic recovery if fire starts to die out')
    parser.add_argument('--min_particles', type=int, default=5,
                        help='Minimum particles before auto-recovery triggers')
    parser.add_argument('--boost_factor', type=float, default=1.5,
                        help='Intensity boost factor for recovery')
    
    # Output
    parser.add_argument('--save', action='store_true', 
                        help='Save simulation to a file')
    parser.add_argument('--output', type=str, default='forest_fire_animation.mp4',
                        help='Output filename for saved simulation')
    
    return parser.parse_args()

# Monkey-patch the FireParticle class for balanced slow spread
original_init = FireParticle.__init__

def custom_init(self, x, y, intensity=1.0, lifetime=15, spread_rate=0.01, random_strength=0.03, 
                intensity_decay=0.97, min_intensity=0.2):
    original_init(self, x, y, intensity, lifetime)
    self.base_spread_rate = spread_rate
    self.random_strength = random_strength
    self.intensity_decay_base = intensity_decay
    self.min_intensity = min_intensity

original_update = FireParticle.update

def custom_update(self, wind_field, terrain, max_width, max_height, dt=0.1):
    # Age the particle
    self.age += dt
    
    # Update velocity based on wind and terrain
    grid_x, grid_y = int(self.position[0]), int(self.position[1])
    
    # Get wind vector at current position
    if 0 <= grid_x < wind_field.shape[0] and 0 <= grid_y < wind_field.shape[1]:
        wind_vector = wind_field[grid_x, grid_y]
        
        # Apply wind force with some randomness
        random_factor = 1.0 + 0.2 * (np.random.random() - 0.5)
        self.velocity += wind_vector * dt * random_factor
    
    # Terrain effect (uphill acceleration)
    if 0 <= grid_x < terrain.shape[0]-1 and 0 <= grid_y < terrain.shape[1]-1:
        # Calculate gradient (slope) of terrain
        dx = terrain[min(grid_x+1, terrain.shape[0]-1), grid_y] - terrain[grid_x, grid_y]
        dy = terrain[grid_x, min(grid_y+1, terrain.shape[1]-1)] - terrain[grid_x, grid_y]
        
        # Fire spreads faster uphill (positive gradient)
        terrain_effect = np.array([dx, dy]) 
        
        # Stronger effect for steeper slopes
        slope_magnitude = np.sqrt(dx**2 + dy**2)
        slope_factor = 1.0 + 2.0 * slope_magnitude
        
        self.velocity += terrain_effect * 0.05 * dt * slope_factor
    
    # Ensure minimal baseline movement even without wind or terrain
    if np.linalg.norm(self.velocity) < self.base_spread_rate:
        # Generate random direction for base spread
        random_direction = np.random.rand(2) * 2 - 1
        random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-8)
        
        # Apply base spread rate in random direction
        self.velocity += random_direction * self.base_spread_rate * dt * 1.2
    
    # Apply random movement with customizable strength
    random_direction = np.random.rand(2) * 2 - 1
    random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-8)
    random_strength = self.random_strength * self.intensity
    self.velocity += random_direction * random_strength * dt
    
    # Limit maximum velocity but ensure minimum velocity
    max_speed = 0.8 * self.intensity
    min_speed = self.base_spread_rate * 0.3
    
    current_speed = np.linalg.norm(self.velocity)
    if current_speed > max_speed:
        self.velocity = (self.velocity / current_speed) * max_speed
    elif current_speed < min_speed and current_speed > 0:
        self.velocity = (self.velocity / current_speed) * min_speed
    
    # Update position
    self.position += self.velocity * dt
    
    # Store path for visualization
    self.path.append((self.position[0], self.position[1]))
    if len(self.path) > self.max_path_length:
        self.path.pop(0)
    
    # Check boundaries
    if not self.is_inbounds(max_width, max_height):
        self.intensity = 0.0
    
    # Balanced intensity decay
    age_factor = min(1.0, self.age / self.lifetime)
    decay_rate = self.intensity_decay_base - (0.05 * age_factor)
    self.intensity *= decay_rate

# Override is_active to use customizable minimum intensity
def custom_is_active(self):
    """Check if the particle is still active using custom minimum intensity"""
    return self.age < self.lifetime and self.intensity > self.min_intensity

# Monkey-patch the ForestFireModel.update method for reliable slow spread
original_model_update = ForestFireModel.update

def custom_model_update(self, dt=0.1):
    """Update the simulation by one time step with balanced slow spread
    
    Args:
        dt (float): Time step size
        
    Returns:
        bool: Whether fire is still active
    """
    new_grid = self.grid.copy()
    new_particles = []
    
    # Update existing particles
    for particle in self.particles:
        particle.update(self.wind_field, self.terrain, self.width, self.height, dt)
        
        if particle.is_active():
            new_particles.append(particle)
            
            # Check for ignition of nearby cells
            x, y = int(particle.position[0]), int(particle.position[1])
            
            # Update max fire spread distance
            if self.ignition_point:
                dist = np.sqrt((x - self.ignition_point[0])**2 + (y - self.ignition_point[1])**2)
                self.fire_spread_distance = max(self.fire_spread_distance, dist)
            
            # Balanced ignition radius
            ignition_radius = int(particle.intensity * 1.8)
            
            for i in range(max(0, x-ignition_radius), min(self.width, x+ignition_radius+1)):
                for j in range(max(0, y-ignition_radius), min(self.height, y+ignition_radius+1)):
                    # Distance from particle to cell
                    distance = np.sqrt((i-particle.position[0])**2 + (j-particle.position[1])**2)
                    
                    if distance <= ignition_radius:
                        # Cell is close enough to potentially ignite
                        if self.grid[i, j] == CellState.FUEL.value:
                            # Balanced ignition probability
                            ignition_prob = (
                                (particle.intensity * 1.0) *
                                (1 - distance/ignition_radius) *
                                self.fuel_types[i, j] *
                                (1 - self.moisture[i, j]) *
                                0.7
                            )
                            
                            # Add base probability
                            ignition_prob = max(self.base_ignition_probability, min(0.7, ignition_prob))
                            
                            # Ignite based on probability
                            if np.random.random() < ignition_prob:
                                new_grid[i, j] = CellState.BURNING.value
                                
                                # Create new particles at a balanced rate
                                if np.random.random() < 0.4:
                                    intensity = np.random.uniform(0.7, 1.0) * particle.intensity
                                    lifetime = np.random.randint(10, 15)
                                    new_particles.append(FireParticle(i, j, intensity, lifetime))
    
    # Update cell states
    for i in range(self.width):
        for j in range(self.height):
            if self.grid[i, j] == CellState.BURNING.value:
                # Each burning cell has a chance to burn out
                if np.random.random() < self.burnout_rate:
                    new_grid[i, j] = CellState.BURNED.value
                    
                # Generate new particles at a balanced rate
                if np.random.random() < self.particle_generation_rate:
                    intensity = np.random.uniform(0.7, 1.0)
                    lifetime = np.random.randint(10, 15)
                    new_particles.append(FireParticle(i, j, intensity, lifetime))
    
    self.grid = new_grid
    self.particles = new_particles
    
    # Auto-recovery if fire is dying out
    if len(new_particles) < self.min_particles and np.sum(self.grid == CellState.BURNING.value) > 0:
        burning_cells = np.where(self.grid == CellState.BURNING.value)
        for _ in range(min(10, len(burning_cells[0]))):
            idx = np.random.randint(0, len(burning_cells[0]))
            i, j = burning_cells[0][idx], burning_cells[1][idx]
            
            # Create boosted particles
            intensity = np.random.uniform(0.9, 1.2) * self.boost_factor
            lifetime = np.random.randint(15, 20)
            new_particles.append(FireParticle(i, j, intensity, lifetime))
    
    # Check if fire has completely died out
    active = len(new_particles) > 0 or np.sum(self.grid == CellState.BURNING.value) > 0
    
    return active

def run_combined_visualization():
    """Combined environment and fire simulation visualization."""
    args = parse_arguments()
    
    # Apply custom init to FireParticle
    def init_override(self, x, y, intensity=1.0, lifetime=args.particle_lifetime):
        custom_init(self, x, y, intensity, lifetime, 
                   spread_rate=args.spread_rate,
                   random_strength=args.random_strength,
                   intensity_decay=args.intensity_decay,
                   min_intensity=args.min_intensity)
    
    FireParticle.__init__ = init_override
    FireParticle.is_active = custom_is_active
    
    # Apply custom update to ForestFireModel
    ForestFireModel.update = custom_model_update
    
    # Create model with specified dimensions
    model = ForestFireModel(args.width, args.height)
    
    # Adjust model parameters for balanced spread
    model.base_ignition_probability = args.ignition_probability
    model.particle_generation_rate = args.particle_generation_rate
    model.initial_particles = args.initial_particles
    model.burnout_rate = args.burnout_rate
    model.min_particles = args.min_particles
    model.boost_factor = args.boost_factor
    
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
    
    # Set up some barriers (e.g., roads or rivers) unless --remove_barriers is specified
    if not args.remove_barriers:
        for i in range(args.width // 3, args.width * 2 // 3):
            model.grid[i, args.height // 2] = CellState.EMPTY.value
    
    # Create figure with 2 subplots side by side
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # 3D terrain view (left)
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Fire simulation (right)
    ax_sim = fig.add_subplot(gs[0, 1])
    
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
    quiver_3d = None
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
        nonlocal fire_points_3d, quiver_3d, frame_count, last_frame_time, fps_history
        
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
        
        # Clear simulation axis and redraw
        ax_sim.clear()
        model.visualize(ax_sim)
        
        # Add FPS counter and parameters
        performance_text = (
            f"FPS: {avg_fps:.1f}\n"
            f"Particles: {len(model.particles)}\n"
            f"Spread Rate: {args.spread_rate}\n"
            f"Ignition Prob: {args.ignition_probability}\n"
            f"Burned: {np.sum(model.grid == CellState.BURNED.value)} cells"
        )
        ax_sim.text(0.02, 0.98, performance_text, transform=ax_sim.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Only update 3D visualization periodically to improve performance
        if frame_count % args.skip_3d_update == 0:
            # First remove old points if they exist
            if fire_points_3d:
                fire_points_3d.remove()
                
            # Update 3D plot with fire particles (limit display count for performance)
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
                
                # Update fire particles on 3D plot
                fire_points_3d = ax_3d.scatter(particle_x, particle_y, particle_z, 
                                             c=colors, s=sizes, marker='o', alpha=0.7)
                
                # Update title with fire information
                ax_3d.set_title(f"3D Terrain - Time: {frame_count}, Active Fires: {len(model.particles)}")
                
                # Update the burned areas on the 3D plot (less frequently)
                if frame_count % (args.skip_3d_update * 2) == 0:
                    # Sample a subset of burned cells
                    burned_indices = np.where(model.grid == CellState.BURNED.value)
                    if burned_indices[0].size > 0:
                        # Take a sample to improve performance
                        sample_size = min(1000, burned_indices[0].size)
                        sample_indices = np.random.choice(burned_indices[0].size, sample_size, replace=False)
                        
                        burned_x = burned_indices[0][sample_indices]
                        burned_y = burned_indices[1][sample_indices]
                        burned_z = [model.terrain[x, y] for x, y in zip(burned_x, burned_y)]
                        
                        # Add black dots for burned areas
                        ax_3d.scatter(burned_x, burned_y, burned_z, 
                                    c='black', s=5, marker='s', alpha=0.3)
        
        # Stop animation if fire is no longer active or max frames reached
        if not active or frame_count >= args.frames:
            print(f"Simulation ended at time step {frame_count}")
            print(f"Final state: {np.sum(model.grid == CellState.FUEL.value)} unburned cells, {np.sum(model.grid == CellState.BURNED.value)} burned cells")
            anim.event_source.stop()
        
        return ax_sim, ax_3d
    
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
