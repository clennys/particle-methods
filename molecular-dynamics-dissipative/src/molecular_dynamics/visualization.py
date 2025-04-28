import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.lines as mlines
import matplotlib.collections as mcollections
from matplotlib.gridspec import GridSpec
import os


class DPDVisualizer:
    COLORS = {0: "blue", 1: "gray", 2: "red", 3: "green"}  # FLUID  # WALL  # A  # B

    TYPE_NAMES = {0: "Fluid", 1: "Wall", 2: "Type A", 3: "Type B"}

    def __init__(self, simulation, update_interval=10):
        self.sim = simulation
        self.update_interval = update_interval
        self.fig = None
        self.ax_sim = None
        self.scatters = {}  # Dict of scatter plots for each particle type
        self.bonds_collection = None
        self.cell_patches = []

    def setup(self):
        """Set up the visualization plot with particle types and bonds"""
        # Create figure with grid layout
        self.fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=self.fig, height_ratios=[1.5, 1])

        # Top row: Main simulation visualization
        self.ax_sim = self.fig.add_subplot(gs[0, :])
        self.ax_sim.set_xlim(0, self.sim.L)
        self.ax_sim.set_ylim(0, self.sim.L)
        self.ax_sim.set_xlabel("x", fontsize=12)
        self.ax_sim.set_ylabel("y", fontsize=12)
        self.ax_sim.set_title(
            f"DPD Simulation - t={self.sim.time:.2f}, T={self.sim.temperature:.3f}",
            fontsize=14,
        )
        self.ax_sim.set_aspect("equal")

        self.draw_cell_grid()

        for type_id in range(4):  # FLUID, WALL, A, B
            indices = np.where(self.sim.types == type_id)[0]
            if len(indices) > 0:
                self.scatters[type_id] = self.ax_sim.scatter(
                    self.sim.positions[indices, 0],
                    self.sim.positions[indices, 1],
                    c=self.COLORS[type_id],
                    alpha=0.7,
                    s=30,
                    label=self.TYPE_NAMES[type_id],
                )

        self.draw_bonds()

        self.ax_sim.legend(loc="upper right", fontsize=10)

        # Temperature plot (left)
        self.ax_temp = self.fig.add_subplot(gs[1, 0])
        self.ax_temp.set_xlabel("Time", fontsize=12)
        self.ax_temp.set_ylabel("Temperature", fontsize=12)
        self.ax_temp.set_title("Temperature vs Time", fontsize=14)
        (self.temp_line,) = self.ax_temp.plot([], [], "r-")
        self.ax_temp.grid(True, alpha=0.3)

        # Velocity profile plot (middle)
        self.ax_velocity = self.fig.add_subplot(gs[1, 1])
        self.ax_velocity.set_xlabel("Velocity (x-component)", fontsize=12)
        self.ax_velocity.set_ylabel("Position (y-direction)", fontsize=12)
        self.ax_velocity.set_title("Velocity Profile", fontsize=14)
        (self.velocity_line,) = self.ax_velocity.plot([], [], "b.-")
        self.ax_velocity.grid(True, alpha=0.3)

        # Density profile plot (right)
        self.ax_density = self.fig.add_subplot(gs[1, 2])
        self.ax_density.set_xlabel("Position (y-direction)", fontsize=12)
        self.ax_density.set_ylabel("Density", fontsize=12)
        self.ax_density.set_title("Density Profile", fontsize=14)

        self.density_lines = {}
        for type_id in range(4):
            if np.any(self.sim.types == type_id):
                (line,) = self.ax_density.plot(
                    [],
                    [],
                    ".-",
                    color=self.COLORS[type_id],
                    label=self.TYPE_NAMES[type_id],
                )
                self.density_lines[type_id] = line

        if self.density_lines:
            self.ax_density.legend(fontsize=10)

        self.ax_density.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.ion()  # Interactive mode on
        plt.show(block=False)

    def draw_cell_grid(self):
        for patch in self.cell_patches:
            if patch in self.ax_sim.patches:
                patch.remove()
        self.cell_patches = []

        cell_size = self.sim.cell_list.cell_size
        num_cells = self.sim.cell_list.num_cells

        for i in range(num_cells):
            for j in range(num_cells):
                rect = Rectangle(
                    (i * cell_size, j * cell_size),
                    cell_size,
                    cell_size,
                    fill=False,
                    linestyle="--",
                    edgecolor="lightgray",
                    alpha=0.2,
                )
                self.ax_sim.add_patch(rect)
                self.cell_patches.append(rect)

    def draw_bonds(self):
        if self.bonds_collection and self.bonds_collection in self.ax_sim.collections:
            self.bonds_collection.remove()

        if not self.sim.bonds:
            return

        segments = []
        for i, j in self.sim.bonds:
            pos_i = self.sim.positions[i]
            pos_j = self.sim.positions[j]

            dr = pos_j - pos_i
            dr = dr - self.sim.L * np.round(dr / self.sim.L)

            segments.append([pos_i, pos_i + dr])

        self.bonds_collection = mcollections.LineCollection(
            segments, colors="black", linewidths=1, alpha=0.5
        )
        self.ax_sim.add_collection(self.bonds_collection)

    def update(self):
        if not self.fig:
            self.setup()

        for type_id, scatter in self.scatters.items():
            indices = np.where(self.sim.types == type_id)[0]
            if len(indices) > 0:
                scatter.set_offsets(self.sim.positions[indices])

        self.draw_bonds()

        self.ax_sim.set_title(
            f"DPD Simulation - t={self.sim.time:.2f}, T={self.sim.temperature:.3f}",
            fontsize=14,
        )

        self.temp_line.set_data(self.sim.time_history, self.sim.temperature_history)
        if len(self.sim.time_history) > 1:
            self.ax_temp.set_xlim(0, max(self.sim.time_history))
            temps = self.sim.temperature_history
            if temps:
                tmin = max(0, min(temps) - 0.1 * abs(min(temps)))
                tmax = max(temps) + 0.1 * abs(max(temps))
                self.ax_temp.set_ylim(tmin, tmax)

        y_bins, vx_profile = self.sim.get_velocity_profile(direction="y", component="x")
        self.velocity_line.set_data(vx_profile, y_bins)

        if len(vx_profile) > 0:
            vx_range = max(abs(np.min(vx_profile)), abs(np.max(vx_profile)))
            if vx_range > 0:
                self.ax_velocity.set_xlim(-vx_range * 1.1, vx_range * 1.1)
            self.ax_velocity.set_ylim(0, self.sim.L)

        for type_id, line in self.density_lines.items():
            y_bins, density = self.sim.get_density_profile(
                direction="y", particle_type=type_id
            )
            line.set_data(y_bins, density)

        if self.density_lines and len(y_bins) > 0:
            self.ax_density.set_xlim(0, self.sim.L)

            max_density = 0
            for type_id in self.density_lines:
                _, density = self.sim.get_density_profile(
                    direction="y", particle_type=type_id
                )
                if len(density) > 0:
                    max_density = max(max_density, np.max(density))

            if max_density > 0:
                self.ax_density.set_ylim(0, max_density * 1.1)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def plot_final_results(self, args):
        final_fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 2, figure=final_fig)

        # Simulation snapshot (top left)
        ax_snapshot = final_fig.add_subplot(gs[0, 0])
        ax_snapshot.set_xlim(0, self.sim.L)
        ax_snapshot.set_ylim(0, self.sim.L)
        ax_snapshot.set_xlabel("x", fontsize=12)
        ax_snapshot.set_ylabel("y", fontsize=12)
        ax_snapshot.set_title(
            f"Final Configuration - t={self.sim.time:.2f}", fontsize=14
        )
        ax_snapshot.set_aspect("equal")

        for type_id in range(4):
            indices = np.where(self.sim.types == type_id)[0]
            if len(indices) > 0:
                ax_snapshot.scatter(
                    self.sim.positions[indices, 0],
                    self.sim.positions[indices, 1],
                    c=self.COLORS[type_id],
                    alpha=0.7,
                    s=30,
                    label=self.TYPE_NAMES[type_id],
                )

        segments = []
        for i, j in self.sim.bonds:
            pos_i = self.sim.positions[i]
            pos_j = self.sim.positions[j]
            dr = pos_j - pos_i
            dr = dr - self.sim.L * np.round(dr / self.sim.L)
            segments.append([pos_i, pos_i + dr])

        if segments:
            bonds_collection = mcollections.LineCollection(
                segments, colors="black", linewidths=1, alpha=0.5
            )
            ax_snapshot.add_collection(bonds_collection)

        ax_snapshot.legend(loc="upper right", fontsize=10)

        ax_temp = final_fig.add_subplot(gs[0, 1])
        ax_temp.plot(self.sim.time_history, self.sim.temperature_history, "r-")
        ax_temp.set_xlabel("Time", fontsize=12)
        ax_temp.set_ylabel("Temperature", fontsize=12)
        ax_temp.set_title("Temperature History", fontsize=14)
        ax_temp.grid(True, alpha=0.3)

        ax_velocity = final_fig.add_subplot(gs[1, 0])
        y_bins, vx_profile = self.sim.get_velocity_profile(
            direction="y", component="x", bins=30
        )
        ax_velocity.plot(vx_profile, y_bins, "b.-")
        ax_velocity.set_xlabel("Velocity (x-component)", fontsize=12)
        ax_velocity.set_ylabel("Position (y-direction)", fontsize=12)
        ax_velocity.set_title("Final Velocity Profile", fontsize=14)
        ax_velocity.grid(True, alpha=0.3)

        ax_density = final_fig.add_subplot(gs[1, 1])
        for type_id in range(4):
            if np.any(self.sim.types == type_id):
                y_bins, density = self.sim.get_density_profile(
                    direction="y", particle_type=type_id, bins=30
                )
                ax_density.plot(
                    y_bins,
                    density,
                    ".-",
                    color=self.COLORS[type_id],
                    label=self.TYPE_NAMES[type_id],
                )

        ax_density.set_xlabel("Position (y-direction)", fontsize=12)
        ax_density.set_ylabel("Density", fontsize=12)
        ax_density.set_title("Final Density Profiles", fontsize=14)
        ax_density.legend(fontsize=10)
        ax_density.grid(True, alpha=0.3)

        ax_energy = final_fig.add_subplot(gs[2, 0])
        ax_energy.plot(
            self.sim.time_history,
            self.sim.kinetic_energy_history,
            "r-",
            label="Kinetic",
        )
        ax_energy.plot(
            self.sim.time_history,
            self.sim.potential_energy_history,
            "g-",
            label="Potential",
        )
        ax_energy.plot(
            self.sim.time_history, self.sim.total_energy_history, "b-", label="Total"
        )
        ax_energy.set_xlabel("Time", fontsize=12)
        ax_energy.set_ylabel("Energy", fontsize=12)
        ax_energy.set_title("Energy History", fontsize=14)
        ax_energy.legend(fontsize=10)
        ax_energy.grid(True, alpha=0.3)

        ax_molecules = final_fig.add_subplot(gs[2, 1])

        if self.sim.molecules:
            molecule_centers_y = []

            for molecule in self.sim.molecules:
                # Calculate center of mass for this molecule
                molecule_positions = np.array(
                    [self.sim.positions[idx] for idx in molecule]
                )
                center = np.mean(molecule_positions, axis=0)
                molecule_centers_y.append(center[1])

            if molecule_centers_y:
                ax_molecules.hist(
                    molecule_centers_y, bins=20, alpha=0.7, color="purple"
                )
                ax_molecules.set_xlabel("Position (y-direction)", fontsize=12)
                ax_molecules.set_ylabel("Number of Molecules", fontsize=12)
                ax_molecules.set_title("Molecule Distribution", fontsize=14)
                ax_molecules.grid(True, alpha=0.3)

        plt.tight_layout()

        plots_dir = os.path.join(args.output, args.scenario)
        final_vis_path = os.path.join(plots_dir, "final_visualization.png")
        plt.savefig(final_vis_path, dpi=150, bbox_inches='tight')

        final_fig.canvas.draw_idle()

    def show(self):
        plt.ioff()
        plt.show()
