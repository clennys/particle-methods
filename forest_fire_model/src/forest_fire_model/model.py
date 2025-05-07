import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import random
from forest_fire_model.particles import *
from scipy.ndimage import gaussian_filter

class ForestFireModel:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Initialize grid
        self.grid = np.zeros((width, height), dtype=int)
        self.grid.fill(CellState.FUEL.value)

        # Fire particles
        self.particles = []
        
        # Environmental factors
        self.wind_field = np.zeros((width, height, 2))  # 2D vector field for wind
        self.terrain = np.zeros((width, height))  # Elevation map
        self.fuel_types = np.ones((width, height))  # Fuel type/density
        self.moisture = np.zeros((width, height))  # Moisture content
        
        # Create a random terrain with some hills and valleys
    def initialize_random_terrain(self):
        self.terrain = np.random.rand(self.width, self.height)
        self.terrain = gaussian_filter(self.terrain, sigma=5)
        
    def set_uniform_wind(self, direction, strength):
        """Set a uniform wind across the entire grid
        Args:
            direction (float): Wind direction in radians (0 = east, pi/2 = north)
            strength (float): Wind strength
        """
        # Convert direction and strength to x,y vector
        wind_x = strength * np.cos(direction)
        wind_y = strength * np.sin(direction)
        
        # Set uniform wind field
        self.wind_field.fill(0)
        for i in range(self.width):
            for j in range(self.height):
                self.wind_field[i, j] = [wind_x, wind_y]
    
    def set_fuel_heterogeneity(self, fuel_types=3):
        """Create patches of different fuel types"""

        # Create random fuel distribution
        self.fuel_types = np.ones((self.width, self.height))
        
        # Add some random fuel patches
        for _ in range(fuel_types):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            radius = np.random.randint(5, 20)
            fuel_value = np.random.uniform(0.5, 2.0)  # Varying fuel densities
            
            for i in range(max(0, x-radius), min(self.width, x+radius)):
                for j in range(max(0, y-radius), min(self.height, y+radius)):
                    if (i-x)**2 + (j-y)**2 <= radius**2:
                        self.fuel_types[i, j] = fuel_value
    
    def set_moisture_gradient(self, base_moisture=0.2):
        """Set moisture with some random variation"""
        # Base moisture level
        self.moisture = np.ones((self.width, self.height)) * base_moisture
        
        # Add random moisture variations
        moisture_variation = np.random.rand(self.width, self.height) * 0.3
        self.moisture += moisture_variation
        
        # Smooth the moisture map
        self.moisture = gaussian_filter(self.moisture, sigma=3)
        
        # Ensure moisture is between 0 and 1
        self.moisture = np.clip(self.moisture, 0, 1)
    
    def ignite(self, x, y):
        """Start a fire at the given location"""
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.grid[x, y] == CellState.FUEL.value:
                self.grid[x, y] = CellState.BURNING.value
                
                # Create initial fire particles
                num_particles = 5  # Number of initial particles
                for _ in range(num_particles):
                    intensity = random.uniform(0.8, 1.2)
                    lifetime = random.randint(8, 12)
                    self.particles.append(FireParticle(x, y, intensity, lifetime))
    
    def update(self, dt=1.0):
        """Update the simulation by one time step"""
        new_grid = self.grid.copy()
        new_particles = []
        
        # Update existing particles
        for particle in self.particles:
            particle.update(self.wind_field, self.terrain, self.width, self.height, dt)
            
            if particle.is_active():
                new_particles.append(particle)
                
                # Check for ignition of nearby cells
                x, y = int(particle.position[0]), int(particle.position[1])
                
                # Ignition radius depends on particle intensity
                ignition_radius = int(particle.intensity * 1.5)
                
                for i in range(max(0, x-ignition_radius), min(self.width, x+ignition_radius+1)):
                    for j in range(max(0, y-ignition_radius), min(self.height, y+ignition_radius+1)):
                        # Distance from particle to cell
                        distance = np.sqrt((i-particle.position[0])**2 + (j-particle.position[1])**2)
                        
                        if distance <= ignition_radius:
                            # Cell is close enough to potentially ignite
                            if self.grid[i, j] == CellState.FUEL.value:
                                # Calculate ignition probability based on:
                                # - Particle intensity
                                # - Distance from particle
                                # - Fuel type
                                # - Moisture content
                                
                                ignition_prob = (
                                    particle.intensity *             # Higher intensity = higher chance
                                    (1 - distance/ignition_radius) * # Closer = higher chance
                                    self.fuel_types[i, j] *          # More fuel = higher chance
                                    (1 - self.moisture[i, j])        # Less moisture = higher chance
                                )
                                
                                # Ignite based on probability
                                if random.random() < ignition_prob:
                                    new_grid[i, j] = CellState.BURNING.value
                                    
                                    # Create new fire particles at newly ignited cells
                                    if random.random() < 0.3:  # Don't create too many particles
                                        intensity = random.uniform(0.7, 1.0) * particle.intensity
                                        lifetime = random.randint(5, 10)
                                        new_particles.append(FireParticle(i, j, intensity, lifetime))
        
        # Update cell states
        for i in range(self.width):
            for j in range(self.height):
                if self.grid[i, j] == CellState.BURNING.value:
                    # Each burning cell has a chance to burn out
                    if random.random() < 0.1:
                        new_grid[i, j] = CellState.BURNED.value
                        
                    # Each burning cell might generate new particles
                    if random.random() < 0.05:
                        intensity = random.uniform(0.8, 1.2)
                        lifetime = random.randint(8, 12)
                        new_particles.append(FireParticle(i, j, intensity, lifetime))
        
        self.grid = new_grid
        self.particles = new_particles
        
        return len(new_particles) > 0  # Return whether fire is still active
    
    def visualize(self, ax=None, show_particles=True):
        """Visualize the current state of the simulation"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a custom colormap for cell states
        colors = ['darkgreen', 'red', 'black', 'lightblue']
        cmap = mcolors.ListedColormap(colors)
        
        # Plot grid
        ax.imshow(self.grid.T, cmap=cmap, origin='lower')
        
        # Plot particles if requested
        if show_particles and self.particles:
            particle_x = [p.position[0] for p in self.particles]
            particle_y = [p.position[1] for p in self.particles]
            intensity = [p.intensity for p in self.particles]
            
            # Scatter plot with intensity determining size
            sizes = [i * 30 for i in intensity]
            ax.scatter(particle_x, particle_y, s=sizes, color='yellow', alpha=0.7)
        
        # Plot terrain contours
        terrain_contour = ax.contour(
            np.arange(self.width), 
            np.arange(self.height), 
            self.terrain.T, 
            levels=10, 
            colors='darkgray', 
            alpha=0.3, 
            linewidths=1.0
        )
        
        # Add some labels
        ax.set_title("Forest Fire Simulation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        return ax

