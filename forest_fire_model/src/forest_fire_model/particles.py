import numpy as np
from enum import Enum

class CellState(Enum):
    FUEL = 0
    BURNING = 1
    BURNED = 2
    EMPTY = 3

class FireParticle:
    def __init__(self, x, y, intensity=1.0, lifetime=10):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.intensity = intensity  # Heat/intensity of the fire
        self.lifetime = lifetime    # How long the particle lives
        self.age = 0                # Current age of particle
    
    def update(self, wind_field, terrain, dt=1.0):
        # Age the particle
        self.age += dt
        
        # Update velocity based on wind and terrain
        # Wind is a 2D vector field with x,y components
        grid_x, grid_y = int(self.position[0]), int(self.position[1])
        
        # Get wind vector at current position
        if 0 <= grid_x < wind_field.shape[0] and 0 <= grid_y < wind_field.shape[1]:
            wind_vector = wind_field[grid_x, grid_y]
            
            # Apply wind force
            self.velocity += wind_vector * dt
        
        # Terrain effect (uphill acceleration)
        if 0 <= grid_x < terrain.shape[0]-1 and 0 <= grid_y < terrain.shape[1]-1:
            # Calculate gradient (slope) of terrain
            dx = terrain[grid_x+1, grid_y] - terrain[grid_x, grid_y]
            dy = terrain[grid_x, grid_y+1] - terrain[grid_x, grid_y]
            
            # Fire moves faster uphill (positive gradient)
            terrain_effect = np.array([dx, dy])
            self.velocity += terrain_effect * 0.1 * dt
        
        self.position += self.velocity * dt
        
        # Decay intensity over time
        # TODO: Refuel the intensity?
        self.intensity *= 0.95
        
    def is_active(self):
        return self.age < self.lifetime and self.intensity > 0.1

