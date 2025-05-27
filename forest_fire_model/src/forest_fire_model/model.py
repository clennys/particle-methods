import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import random
from forest_fire_model.particles import *
from scipy.ndimage import gaussian_filter


class ForestFireModel:
    def __init__(
        self,
        width,
        height,
        spread_rate=0.01,
        random_strength=0.03,
        intensity_decay=0.97,
        min_intensity=0.2,
        ignition_probability=0.03,
        particle_generation_rate=0.03,
        initial_particles=15,
        burnout_rate=0.03,
        particle_lifetime=15,
    ):
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

        # Statistics tracking
        self.fire_spread_distance = (
            0  # Track max distance fire has spread from ignition
        )
        self.ignition_point = None

        # Balanced spread parameters
        self.spread_rate = spread_rate
        self.random_strength = random_strength
        self.intensity_decay = intensity_decay
        self.min_intensity = min_intensity
        self.base_ignition_probability = ignition_probability
        self.particle_generation_rate = particle_generation_rate
        self.initial_particles = initial_particles
        self.burnout_rate = burnout_rate
        self.particle_lifetime = particle_lifetime

    def initialize_random_terrain(self, smoothness=5):
        """Create a random terrain with hills and valleys

        Args:
            smoothness (int): Higher values create smoother terrain (default: 5)
        """
        self.terrain = np.random.rand(self.width, self.height)
        self.terrain = gaussian_filter(self.terrain, sigma=smoothness)

    def set_uniform_wind(self, direction, strength):
        """Set a uniform wind across the entire grid

        Args:
            direction (float): Wind direction in radians (0 = east, pi/2 = north)
            strength (float): Wind strength (0-1 range recommended)
        """
        # Convert direction and strength to x,y vector
        wind_x = strength * np.cos(direction)
        wind_y = strength * np.sin(direction)

        # Set uniform wind field
        self.wind_field.fill(0)
        for i in range(self.width):
            for j in range(self.height):
                self.wind_field[i, j] = [wind_x, wind_y]

    def set_variable_wind(self, base_direction, base_strength, variability=0.2):
        """Set a variable wind field with some randomness

        Args:
            base_direction (float): Base wind direction in radians
            base_strength (float): Base wind strength
            variability (float): Amount of random variation (0-1)
        """
        # Create random variations
        dir_var = np.random.rand(self.width, self.height) * variability * np.pi - (
            variability * np.pi / 2
        )
        str_var = np.random.rand(self.width, self.height) * variability

        # Smooth the variations
        dir_var = gaussian_filter(dir_var, sigma=5)
        str_var = gaussian_filter(str_var, sigma=5)

        # Apply to wind field
        for i in range(self.width):
            for j in range(self.height):
                direction = base_direction + dir_var[i, j]
                strength = base_strength * (1 + str_var[i, j])

                wind_x = strength * np.cos(direction)
                wind_y = strength * np.sin(direction)
                self.wind_field[i, j] = [wind_x, wind_y]

    def set_fuel_heterogeneity(self, fuel_types=3):
        """Create patches of different fuel types

        Args:
            fuel_types (int): Number of different fuel patches to create
        """
        # Create random fuel distribution
        self.fuel_types = np.ones((self.width, self.height))

        # Add some random fuel patches
        for _ in range(fuel_types):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            radius = np.random.randint(5, 20)
            fuel_value = np.random.uniform(0.5, 2.0)  # Varying fuel densities

            for i in range(max(0, x - radius), min(self.width, x + radius)):
                for j in range(max(0, y - radius), min(self.height, y + radius)):
                    if (i - x) ** 2 + (j - y) ** 2 <= radius**2:
                        self.fuel_types[i, j] = fuel_value

    def set_moisture_gradient(self, base_moisture=0.2):
        """Set moisture with some random variation

        Args:
            base_moisture (float): Base moisture level (0-1)
        """
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
        """Start a fire at the given location with radial spread

        Args:
            x (int): X coordinate to ignite
            y (int): Y coordinate to ignite
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.grid[x, y] == CellState.FUEL.value:
                self.grid[x, y] = CellState.BURNING.value
                self.ignition_point = (x, y)

                # Create particles in all directions using balanced parameters
                num_directions = (
                    self.initial_particles
                )  # Use configurable initial particles
                for i in range(num_directions):
                    angle = 2 * np.pi * i / num_directions
                    direction_x = np.cos(angle)
                    direction_y = np.sin(angle)

                    # Create particle with balanced spread parameters
                    intensity = random.uniform(0.8, 1.2)
                    new_particle = FireParticle(
                        x,
                        y,
                        intensity,
                        self.particle_lifetime,
                        spread_rate=self.spread_rate,
                        random_strength=self.random_strength,
                        intensity_decay=self.intensity_decay,
                        min_intensity=self.min_intensity,
                    )

                    # Set initial velocity based on direction
                    new_particle.velocity = np.array([direction_x, direction_y]) * 0.2

                    self.particles.append(new_particle)

    def update(self, dt=0.1):
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
                    dist = np.sqrt(
                        (x - self.ignition_point[0]) ** 2
                        + (y - self.ignition_point[1]) ** 2
                    )
                    self.fire_spread_distance = max(self.fire_spread_distance, dist)

                # Balanced ignition radius
                ignition_radius = int(particle.intensity * 1.8)

                for i in range(
                    max(0, x - ignition_radius),
                    min(self.width, x + ignition_radius + 1),
                ):
                    for j in range(
                        max(0, y - ignition_radius),
                        min(self.height, y + ignition_radius + 1),
                    ):
                        # Distance from particle to cell
                        distance = np.sqrt(
                            (i - particle.position[0]) ** 2
                            + (j - particle.position[1]) ** 2
                        )

                        if distance <= ignition_radius:
                            # Cell is close enough to potentially ignite
                            if self.grid[i, j] == CellState.FUEL.value:
                                # Balanced ignition probability
                                ignition_prob = (
                                    (particle.intensity * 1.0)
                                    * (1 - distance / ignition_radius)
                                    * self.fuel_types[i, j]
                                    * (1 - self.moisture[i, j])
                                    * 0.7
                                )

                                # Add base probability
                                ignition_prob = max(
                                    self.base_ignition_probability,
                                    min(0.7, ignition_prob),
                                )

                                # Ignite based on probability
                                if np.random.random() < ignition_prob:
                                    new_grid[i, j] = CellState.BURNING.value

                                    # Create new particles at a balanced rate
                                    if np.random.random() < 0.4:
                                        intensity = (
                                            np.random.uniform(0.7, 1.0)
                                            * particle.intensity
                                        )
                                        new_particle = FireParticle(
                                            i,
                                            j,
                                            intensity,
                                            self.particle_lifetime,
                                            spread_rate=self.spread_rate,
                                            random_strength=self.random_strength,
                                            intensity_decay=self.intensity_decay,
                                            min_intensity=self.min_intensity,
                                        )
                                        new_particles.append(new_particle)

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
                        new_particle = FireParticle(
                            i,
                            j,
                            intensity,
                            self.particle_lifetime,
                            spread_rate=self.spread_rate,
                            random_strength=self.random_strength,
                            intensity_decay=self.intensity_decay,
                            min_intensity=self.min_intensity,
                        )
                        new_particles.append(new_particle)

        self.grid = new_grid
        self.particles = new_particles

        # Check if fire has completely died out
        active = (
            len(new_particles) > 0 or np.sum(self.grid == CellState.BURNING.value) > 0
        )

        return active

    def visualize(
        self, ax=None, show_particles=True, show_terrain=True, show_fuel=False
    ):
        """Visualize the current state of the simulation

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on (creates new if None)
            show_particles (bool): Whether to show fire particles
            show_terrain (bool): Whether to show terrain contours
            show_fuel (bool): Whether to show fuel density as contours

        Returns:
            matplotlib.axes.Axes: The plotting axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Create a custom colormap for cell states
        colors = ["darkgreen", "red", "black", "lightblue"]
        cmap = mcolors.ListedColormap(colors)

        # Plot grid
        img = ax.imshow(
            self.grid.T,
            cmap=cmap,
            origin="lower",
            extent=[0, self.width, 0, self.height],
        )

        # Add a colorbar with labels
        if not hasattr(self, "_colorbar_added"):
            cbar = plt.colorbar(img, ax=ax, ticks=[0.4, 1.2, 2.1, 2.9])
            cbar.ax.set_yticklabels(["Fuel", "Burning", "Burned", "Empty"])
            self._colorbar_added = True

        # Plot particles if requested
        if show_particles and self.particles:
            particle_x = [p.position[0] for p in self.particles]
            particle_y = [p.position[1] for p in self.particles]
            intensity = [p.intensity for p in self.particles]

            # Scatter plot with intensity determining size
            sizes = [i * 30 for i in intensity]
            ax.scatter(particle_x, particle_y, s=sizes, color="yellow", alpha=0.7)

        # Plot terrain contours
        if show_terrain:
            terrain_contour = ax.contour(
                np.arange(self.width),
                np.arange(self.height),
                self.terrain.T,
                levels=10,
                colors="darkgray",
                alpha=0.3,
                linewidths=1.0,
            )

            # Add contour labels (only occasionally)
            if (
                random.random() < 0.1
            ):  # Only add labels 10% of the time to avoid crowding
                ax.clabel(terrain_contour, inline=True, fontsize=8, fmt="%.1f")

        # Plot fuel density as contours
        if show_fuel:
            fuel_contour = ax.contour(
                np.arange(self.width),
                np.arange(self.height),
                self.fuel_types.T,
                levels=5,
                colors="green",
                alpha=0.4,
                linewidths=1.0,
            )

        # Add statistics and info
        burned_percent = (
            np.sum(self.grid == CellState.BURNED.value)
            / (self.width * self.height)
            * 100
        )
        burning_count = np.sum(self.grid == CellState.BURNING.value)
        fuel_left = np.sum(self.grid == CellState.FUEL.value)

        info_text = (
            f"Fire spread: {self.fire_spread_distance:.1f} units\n"
            f"Burned: {burned_percent:.1f}%\n"
            f"Active fires: {burning_count}\n"
            f"Fuel remaining: {fuel_left}"
        )

        # Position the text box in the bottom right corner
        ax.text(
            0.98,
            0.02,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        # Add grid lines
        ax.grid(which="both", color="gray", linestyle="-", linewidth=0.5, alpha=0.2)

        # Add labels
        ax.set_title("Forest Fire Simulation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        return ax

    def visualize_environment(self, ax=None):
        """Create a visualization of the environmental factors

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on (creates new if None)

        Returns:
            matplotlib.axes.Axes: The plotting axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Create a terrain heatmap
        terrain_img = ax.imshow(
            self.terrain.T,
            cmap="terrain",
            origin="lower",
            alpha=0.7,
            extent=[0, self.width, 0, self.height],
        )
        plt.colorbar(terrain_img, ax=ax, label="Elevation")

        # Show moisture as contours
        moisture_contour = ax.contour(
            np.arange(self.width),
            np.arange(self.height),
            self.moisture.T,
            levels=5,
            colors="blue",
            alpha=0.4,
            linewidths=1.0,
        )
        ax.clabel(moisture_contour, inline=True, fontsize=8, fmt="%.1f")

        # Show fuel density
        fuel_contour = ax.contour(
            np.arange(self.width),
            np.arange(self.height),
            self.fuel_types.T,
            levels=5,
            colors="green",
            alpha=0.5,
            linewidths=1.0,
        )
        ax.clabel(fuel_contour, inline=True, fontsize=8, fmt="%.1f")

        # Visualize wind field with downsampled arrows
        skip = max(1, self.width // 20)  # Show fewer arrows for clarity
        x_points = np.arange(0, self.width, skip)
        y_points = np.arange(0, self.height, skip)
        X, Y = np.meshgrid(x_points, y_points)

        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i, x in enumerate(x_points):
            for j, y in enumerate(y_points):
                if x < self.width and y < self.height:
                    wind_vec = self.wind_field[int(x), int(y)]
                    U[j, i] = wind_vec[0]
                    V[j, i] = wind_vec[1]

        # Scale arrows for better visibility
        arrow_scale = 30
        quiver = ax.quiver(X, Y, U, V, color="black", scale=arrow_scale)

        # Add a key for scale reference
        ax.quiverkey(
            quiver, 0.9, 0.95, 0.5, "Wind: 0.5", labelpos="E", coordinates="figure"
        )

        # Add barriers from the grid
        barrier_x, barrier_y = [], []
        for i in range(self.width):
            for j in range(self.height):
                if self.grid[i, j] == CellState.EMPTY.value:
                    barrier_x.append(i)
                    barrier_y.append(j)

        if barrier_x:
            ax.scatter(barrier_x, barrier_y, c="gray", marker="s", s=20)

        # Add labels
        ax.set_title("Environmental Factors")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(which="both", color="gray", linestyle="-", linewidth=0.5, alpha=0.2)

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                color="black",
                marker=">",
                linestyle="-",
                markersize=8,
                label="Wind Direction",
            ),
            Line2D(
                [0], [0], color="blue", linestyle="-", markersize=8, label="Moisture"
            ),
            Line2D(
                [0],
                [0],
                color="green",
                linestyle="-",
                markersize=8,
                label="Fuel Density",
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                marker="s",
                linestyle="",
                markersize=8,
                label="Barriers",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper left")

        return ax
