"""
Visualization module for Lennard-Jones simulation with two-row layout
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


class Visualizer:
    """
    Class for visualizing the Lennard-Jones simulation with a two-row layout
    """

    def __init__(self, simulation, update_interval=10):
        """
        Initialize visualizer.

        Parameters:
        -----------
        simulation : Simulation
            Reference to the simulation object
        update_interval : int
            Interval between visualization updates
        """
        self.sim = simulation
        self.update_interval = update_interval
        self.fig = None
        self.ax_sim = None
        self.scatter = None

    def setup(self):
        """Set up the visualization plot with two-row layout"""
        # Create figure with two rows
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle("Lennard-Jones 2D Simulation", fontsize=18)
        
        # Create grid layout: 2 rows, 3 columns
        gs = GridSpec(2, 3, figure=self.fig, height_ratios=[1.5, 1])
        
        # Top row: Main simulation visualization (larger)
        self.ax_sim = self.fig.add_subplot(gs[0, :])
        self.ax_sim.set_xlim(0, self.sim.L)
        self.ax_sim.set_ylim(0, self.sim.L)
        self.ax_sim.set_xlabel("x", fontsize=12)
        self.ax_sim.set_ylabel("y", fontsize=12)
        self.ax_sim.set_title(f"N={self.sim.N}, T={self.sim.temperature:.2f}", fontsize=14)
        self.ax_sim.set_aspect("equal")
        
        # Draw particles with smaller size
        self.scatter = self.ax_sim.scatter(
            self.sim.positions[:, 0], self.sim.positions[:, 1], c="b", alpha=0.7, s=30
        )
        
        # Bottom row: Three plots side by side
        
        # Energy plot (left)
        self.ax_energy = self.fig.add_subplot(gs[1, 0])
        self.ax_energy.set_xlabel("Time", fontsize=12)
        self.ax_energy.set_ylabel("Energy", fontsize=12)
        self.ax_energy.set_title("Energy vs Time", fontsize=14)
        (self.kinetic_line,) = self.ax_energy.plot([], [], "r-", label="Kinetic")
        (self.potential_line,) = self.ax_energy.plot([], [], "g-", label="Potential")
        (self.total_line,) = self.ax_energy.plot([], [], "b-", label="Total")
        self.ax_energy.legend(fontsize=10)
        self.ax_energy.grid(True, alpha=0.3)
        
        # Temperature plot (middle)
        self.ax_temp = self.fig.add_subplot(gs[1, 1])
        self.ax_temp.set_xlabel("Time", fontsize=12)
        self.ax_temp.set_ylabel("Temperature", fontsize=12)
        self.ax_temp.set_title("Temperature vs Time", fontsize=14)
        (self.temp_line,) = self.ax_temp.plot([], [], "r-")
        self.ax_temp.grid(True, alpha=0.3)
        
        # RDF plot (right)
        self.ax_rdf = self.fig.add_subplot(gs[1, 2])
        self.ax_rdf.set_xlabel("r", fontsize=12)
        self.ax_rdf.set_ylabel("g(r)", fontsize=12)
        self.ax_rdf.set_title("Radial Distribution Function", fontsize=14)
        self.ax_rdf.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.ion()  # Interactive mode on
        plt.show(block=False)

    def update(self):
        """Update the visualization with current simulation state"""
        if not self.fig:
            self.setup()

        # Update particle positions
        self.scatter.set_offsets(self.sim.positions)

        # Update plot title with current temperature
        self.ax_sim.set_title(f"N={self.sim.N}, T={self.sim.temperature:.3f}", fontsize=14)

        # Update energy plots
        times = self.sim.time_history
        self.kinetic_line.set_data(times, self.sim.kinetic_energy_history)
        self.potential_line.set_data(times, self.sim.potential_energy_history)
        self.total_line.set_data(times, self.sim.total_energy_history)

        # Adjust y-axis limits for energy plot
        if len(times) > 1:
            self.ax_energy.set_xlim(0, max(times))

            all_energies = np.concatenate(
                [
                    self.sim.kinetic_energy_history,
                    self.sim.potential_energy_history,
                    self.sim.total_energy_history,
                ]
            )

            ymin = min(all_energies) - 0.1 * abs(min(all_energies))
            ymax = max(all_energies) + 0.1 * abs(max(all_energies))
            self.ax_energy.set_ylim(ymin, ymax)

        # Update temperature plot
        self.temp_line.set_data(times, self.sim.temperature_history)

        if len(times) > 1:
            self.ax_temp.set_xlim(0, max(times))

            temps = self.sim.temperature_history
            if temps:
                tmin = min(temps) - 0.1 * abs(min(temps))
                tmax = max(temps) + 0.1 * abs(max(temps))
                self.ax_temp.set_ylim(max(0, tmin), tmax)

        # Draw updates
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def plot_rdf(self):
        """Plot the radial distribution function"""
        if hasattr(self.sim, "rdf") and hasattr(self.sim, "rdf_r"):
            self.ax_rdf.clear()
            self.ax_rdf.plot(self.sim.rdf_r, self.sim.rdf, "b-")
            self.ax_rdf.set_xlabel("r", fontsize=12)
            self.ax_rdf.set_ylabel("g(r)", fontsize=12)
            self.ax_rdf.set_title("Radial Distribution Function", fontsize=14)
            self.ax_rdf.grid(True, alpha=0.3)

            # Fix RDF y-axis limits
            if len(self.sim.rdf) > 0:
                ymax = max(3.0, np.max(self.sim.rdf) * 1.1)
                self.ax_rdf.set_ylim(0, ymax)

            self.fig.canvas.draw_idle()

    def plot_energies(self):
        """Plot the final energy data"""
        times = self.sim.time_history

        if not times:
            return

        # Adjust all plots for final view
        self.ax_energy.clear()
        self.ax_energy.plot(
            times, self.sim.kinetic_energy_history, "r-", label="Kinetic"
        )
        self.ax_energy.plot(
            times, self.sim.potential_energy_history, "g-", label="Potential"
        )
        self.ax_energy.plot(times, self.sim.total_energy_history, "b-", label="Total")
        self.ax_energy.set_xlabel("Time", fontsize=12)
        self.ax_energy.set_ylabel("Energy", fontsize=12)
        self.ax_energy.set_title("Energy vs Time", fontsize=14)
        self.ax_energy.legend(fontsize=10)
        self.ax_energy.grid(True, alpha=0.3)

        self.ax_temp.clear()
        self.ax_temp.plot(times, self.sim.temperature_history, "r-")
        self.ax_temp.set_xlabel("Time", fontsize=12)
        self.ax_temp.set_ylabel("Temperature", fontsize=12)
        self.ax_temp.set_title("Temperature vs Time", fontsize=14)
        self.ax_temp.grid(True, alpha=0.3)

        self.fig.canvas.draw_idle()

    def show(self):
        """Show the plots and keep window open"""
        plt.ioff()
        plt.show()
