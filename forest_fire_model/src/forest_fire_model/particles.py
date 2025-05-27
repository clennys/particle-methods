import numpy as np
import random
from enum import Enum


class CellState(Enum):
    FUEL = 0  # Unburned vegetation
    BURNING = 1  # Currently burning
    BURNED = 2  # Already burned
    EMPTY = 3  # No fuel (e.g., roads, water)


class FireParticle:
    def __init__(
        self,
        x,
        y,
        intensity=1.0,
        lifetime=15,
        spread_rate=0.01,
        random_strength=0.03,
        intensity_decay=0.97,
        min_intensity=0.2,
    ):
        """Initialize a fire particle with balanced spread parameters

        Args:
            x (float): Initial x position
            y (float): Initial y position
            intensity (float): Fire intensity/heat (affects spread radius)
            lifetime (int): How many time steps the particle lives
            spread_rate (float): Base fire spread rate
            random_strength (float): Strength of random movement
            intensity_decay (float): Rate at which intensity decays
            min_intensity (float): Minimum intensity before particle dies
        """
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.intensity = intensity
        self.lifetime = lifetime
        self.age = 0

        # Balanced spread parameters
        self.base_spread_rate = spread_rate
        self.random_strength = random_strength
        self.intensity_decay_base = intensity_decay
        self.min_intensity = min_intensity

        # Track the path for visualization
        self.path = [(x, y)]
        self.max_path_length = 10

    def is_active(self):
        """Check if the particle is still active using custom minimum intensity"""
        return self.age < self.lifetime and self.intensity > self.min_intensity

    def is_inbounds(self, width, height):
        """Check if the particle is within the grid boundaries"""
        return 0 <= self.position[0] < width and 0 <= self.position[1] < height

    def update(self, wind_field, terrain, max_width, max_height, dt=0.1):
        """Update the particle with balanced fire spread behavior"""
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
        if 0 <= grid_x < terrain.shape[0] - 1 and 0 <= grid_y < terrain.shape[1] - 1:
            # Calculate gradient (slope) of terrain
            dx = (
                terrain[min(grid_x + 1, terrain.shape[0] - 1), grid_y]
                - terrain[grid_x, grid_y]
            )
            dy = (
                terrain[grid_x, min(grid_y + 1, terrain.shape[1] - 1)]
                - terrain[grid_x, grid_y]
            )

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
            random_direction = random_direction / (
                np.linalg.norm(random_direction) + 1e-8
            )

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

    def draw(self, ax, show_path=True):
        """Draw the particle on the given axes

        Args:
            ax (matplotlib.axes.Axes): Axes to draw on
            show_path (bool): Whether to show the particle's path
        """
        # Draw the particle as a point
        ax.scatter(
            self.position[0],
            self.position[1],
            color="yellow",
            s=self.intensity * 30,  # Size based on intensity
            alpha=min(1.0, self.intensity),
        )

        # Optionally draw the path
        if show_path and len(self.path) > 1:
            path_x, path_y = zip(*self.path)
            # Alpha increases for more recent positions
            alphas = np.linspace(0.1, 0.5, len(path_x))

            for i in range(len(path_x) - 1):
                ax.plot(
                    [path_x[i], path_x[i + 1]],
                    [path_y[i], path_y[i + 1]],
                    color="orange",
                    alpha=alphas[i],
                    linewidth=1,
                )

        # Draw velocity vector
        if np.linalg.norm(self.velocity) > 0.01:
            vec_length = np.linalg.norm(self.velocity) * 2
            norm_vel = self.velocity / (np.linalg.norm(self.velocity) + 1e-8)
            ax.arrow(
                self.position[0],
                self.position[1],
                norm_vel[0] * vec_length,
                norm_vel[1] * vec_length,
                head_width=0.3,
                head_length=0.5,
                fc="red",
                ec="red",
                alpha=0.5,
            )
