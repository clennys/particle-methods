import numpy as np
import random
from enum import Enum

class CellState(Enum):
    FUEL = 0      # Unburned vegetation
    BURNING = 1   # Currently burning
    BURNED = 2    # Already burned
    EMPTY = 3     # No fuel (e.g., roads, water)

class FireParticle:
    def __init__(self, x, y, intensity=1.0, lifetime=10):
        """Initialize a fire particle
        
        Args:
            x (float): Initial x position
            y (float): Initial y position
            intensity (float): Fire intensity/heat (affects spread radius)
            lifetime (int): How many time steps the particle lives
        """
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.intensity = intensity  # Heat/intensity of the fire
        self.lifetime = lifetime    # How long the particle lives
        self.age = 0                # Current age of particle
        
        # Track the path for visualization
        self.path = [(x, y)]
        self.max_path_length = 10  # Maximum number of positions to remember

    def is_active(self):
        """Check if the particle is still active"""
        return self.age < self.lifetime and self.intensity > 0.1

    def is_inbounds(self, width, height):
        """Check if the particle is within the grid boundaries"""
        return 0 <= self.position[0] < width and 0 <= self.position[1] < height
    
    def update(self, wind_field, terrain, max_width, max_height, dt=1.0):
        """Update the particle position and state
        
        Args:
            wind_field (numpy.ndarray): 3D array of wind vectors
            terrain (numpy.ndarray): 2D array of terrain heights
            max_width (int): Grid width
            max_height (int): Grid height
            dt (float): Time step size
        """
        # Age the particle
        self.age += dt
        
        # Update velocity based on wind and terrain
        grid_x, grid_y = int(self.position[0]), int(self.position[1])
        
        # Get wind vector at current position
        if 0 <= grid_x < wind_field.shape[0] and 0 <= grid_y < wind_field.shape[1]:
            wind_vector = wind_field[grid_x, grid_y]
            
            # Apply wind force with some randomness
            random_factor = 1.0 + 0.2 * (random.random() - 0.5)  # +/- 10% randomness
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
            slope_factor = 1.0 + 2.0 * slope_magnitude  # Amplify effect of steep slopes
            
            self.velocity += terrain_effect * 0.1 * dt * slope_factor
        
        # Apply a small random movement
        random_direction = np.random.rand(2) * 2 - 1  # Random direction vector
        random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-8)  # Normalize
        random_strength = 0.05 * self.intensity  # Stronger fire, more erratic
        self.velocity += random_direction * random_strength * dt
        
        # Limit maximum velocity
        max_speed = 1.5 * self.intensity  # Faster speed for higher intensity
        current_speed = np.linalg.norm(self.velocity)
        if current_speed > max_speed:
            self.velocity = (self.velocity / current_speed) * max_speed
        
        # Update position
        old_position = self.position.copy()
        self.position += self.velocity * dt
        
        # Store path for visualization (keep last N positions)
        self.path.append((self.position[0], self.position[1]))
        if len(self.path) > self.max_path_length:
            self.path.pop(0)
        
        # Check boundaries and either stop or wrap around
        if not self.is_inbounds(max_width, max_height):
            # Option 1: Reduce intensity (effectively killing the particle when out of bounds)
            self.intensity = 0.0
            
            # Option 2: Implement periodic boundary conditions (uncomment to enable)
            # self.position[0] = self.position[0] % max_width
            # self.position[1] = self.position[1] % max_height
        
        # Decay intensity over time
        # Intensity decreases faster with age
        age_factor = min(1.0, self.age / self.lifetime)
        decay_rate = 0.95 - (0.1 * age_factor)  # Decay rate increases with age
        self.intensity *= decay_rate
        
    def draw(self, ax, show_path=True):
        """Draw the particle on the given axes
        
        Args:
            ax (matplotlib.axes.Axes): Axes to draw on
            show_path (bool): Whether to show the particle's path
        """
        # Draw the particle as a point
        ax.scatter(self.position[0], self.position[1], 
                 color='yellow', 
                 s=self.intensity * 30,  # Size based on intensity
                 alpha=min(1.0, self.intensity))
        
        # Optionally draw the path
        if show_path and len(self.path) > 1:
            path_x, path_y = zip(*self.path)
            # Alpha increases for more recent positions
            alphas = np.linspace(0.1, 0.5, len(path_x))
            
            for i in range(len(path_x) - 1):
                ax.plot([path_x[i], path_x[i+1]], [path_y[i], path_y[i+1]], 
                       color='orange', alpha=alphas[i], linewidth=1)
        
        # Draw velocity vector
        if np.linalg.norm(self.velocity) > 0.01:
            vec_length = np.linalg.norm(self.velocity) * 2
            norm_vel = self.velocity / (np.linalg.norm(self.velocity) + 1e-8)
            ax.arrow(self.position[0], self.position[1], 
                    norm_vel[0] * vec_length, norm_vel[1] * vec_length,
                    head_width=0.3, head_length=0.5, fc='red', ec='red', alpha=0.5)
