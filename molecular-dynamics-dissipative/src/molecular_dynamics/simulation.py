import numpy as np
from cell_list import CellList
from forces import compute_dpd_forces, compute_bond_forces, apply_body_force


class DPDSimulation:
    FLUID = 0
    WALL = 1
    TYPE_A = 2
    TYPE_B = 3

    TYPE_MAP = {"F": FLUID, "W": WALL, "A": TYPE_A, "B": TYPE_B}

    def __init__(
        self,
        L=15.0,
        density=4.0,
        dt=0.01,
        rc=1.0,
        sigma=3.0,
        gamma=4.5,
        kBT=1.0,
        a_matrix=None,
        K_S=100.0,
        r_S=0.1,
        body_force=0.0,
    ):

        self.L = L
        self.density = density
        self.dt = dt
        self.rc = rc
        self.sigma = sigma
        self.gamma = gamma
        self.kBT = kBT

        if abs(self.sigma**2 - 2 * self.gamma * self.kBT) > 1e-10:
            print(
                "Warning: DPD thermostat relation is not satisfied: sigma^2 = 2*gamma*kBT"
            )
            print(f"  sigma^2 = {self.sigma**2}, 2*gamma*kBT = {2*self.gamma*self.kBT}")
            print(
                f"  For current gamma={self.gamma} and kBT={self.kBT}, sigma should be {np.sqrt(2*self.gamma*self.kBT)}"
            )
        else:
            print("DPD thermostat relation is satisfied: sigma^2 = 2*gamma*kBT")

        self.K_S = K_S
        self.r_S = r_S
        self.body_force = body_force

        self.N_total = int(self.density * self.L**2)
        print(f"Total number of particles: {self.N_total}")

        if a_matrix is None:
            self.a_matrix = np.full((4, 4), 25.0)
        else:
            self.a_matrix = a_matrix

        self.positions = np.zeros((self.N_total, 2))
        self.velocities = np.zeros((self.N_total, 2))
        self.forces = np.zeros((self.N_total, 2))
        self.types = np.full(self.N_total, self.FLUID, dtype=int)

        self.bonds = []  # List of (i, j) pairs of bonded particles
        self.molecules = []  # List of sets of particle indices for each molecule

        self.wall_velocity = np.zeros(2)

        self.cell_list = CellList(L, rc)

        self.kinetic_energy = 0.0
        self.potential_energy = 0.0
        self.bond_energy = 0.0
        self.temperature = 0.0

        self.time = 0.0
        self.time_history = []
        self.temperature_history = []
        self.kinetic_energy_history = []
        self.potential_energy_history = []
        self.total_energy_history = []

        self.velocity_profile = None
        self.velocity_profile_bins = None

        self.density_profile = None
        self.density_profile_bins = None

        self.initialize_random_positions()

    def initialize_random_positions(self):
        self.positions = np.random.uniform(0, self.L, (self.N_total, 2))

        self.velocities = np.zeros((self.N_total, 2))

        self.update_cell_list()

    def update_cell_list(self):
        self.cell_list.update(self.positions)

    def create_chain_molecules(self, num_chains):
        chain_molecules = []
        new_bonds = []

        available_indices = np.where(self.types == self.FLUID)[0]

        if len(available_indices) < num_chains * 7:
            raise ValueError(
                f"Not enough fluid particles to create {num_chains} chains of 7 particles each"
            )

        for i in range(num_chains):
            # Select 7 particles for this chain
            chain_indices = available_indices[i * 7 : (i + 1) * 7]

            chain_molecules.append(set(chain_indices))

            # Set particle types (2 A followed by 5 B)
            self.types[chain_indices[0:2]] = self.TYPE_A
            self.types[chain_indices[2:7]] = self.TYPE_B

            # Create bonds between adjacent particles
            for j in range(6):
                new_bonds.append((int(chain_indices[j]), int(chain_indices[j + 1])))

            base_pos = np.random.uniform(self.rc, self.L - self.rc, 2)
            for j, idx in enumerate(chain_indices):
                offset = np.array([j * self.r_S * 1.2, 0]) + np.random.uniform(
                    -0.1, 0.1, 2
                )
                new_pos = base_pos + offset
                self.positions[idx] = np.mod(new_pos, self.L)

        self.molecules.extend(chain_molecules)

        self.bonds.extend(new_bonds)

        self.update_cell_list()

        print(f"Created {num_chains} chain molecules")

    def create_ring_molecules(self, num_rings, ring_size=9):
        ring_molecules = []
        new_bonds = []

        available_indices = np.where(self.types == self.FLUID)[0]

        if len(available_indices) < num_rings * ring_size:
            raise ValueError(
                f"Not enough fluid particles to create {num_rings} rings of {ring_size} particles each"
            )

        for i in range(num_rings):
            ring_indices = available_indices[i * ring_size : (i + 1) * ring_size]

            ring_molecules.append(set(ring_indices))

            self.types[ring_indices] = self.TYPE_A

            for j in range(ring_size):
                next_j = (j + 1) % ring_size
                new_bonds.append((int(ring_indices[j]), int(ring_indices[next_j])))

            center = np.random.uniform(2 * self.rc, self.L - 2 * self.rc, 2)
            radius = self.r_S * ring_size / (2 * np.pi)

            for j, idx in enumerate(ring_indices):
                angle = 2 * np.pi * j / ring_size
                offset = radius * np.array([np.cos(angle), np.sin(angle)])
                offset += np.random.uniform(-0.05, 0.05, 2)  # Small random perturbation
                new_pos = center + offset
                self.positions[idx] = np.mod(new_pos, self.L)

        self.molecules.extend(ring_molecules)
        self.bonds.extend(new_bonds)

        self.update_cell_list()

        print(f"Created {num_rings} ring molecules with {ring_size} particles each")

    def create_walls(self, thickness=None, positions=None):
        if thickness is None:
            thickness = self.rc

        if positions is None:
            positions = ["y=0", "y=L"]

        if positions == "x":
            positions = ["x=0", "x=L"]
        elif positions == "y":
            positions = ["y=0", "y=L"]

        # Convert fluid particles to wall particles in the wall regions
        for wall_pos in positions:
            if wall_pos == "x=0":
                wall_indices = np.where(
                    (self.types == self.FLUID) & (self.positions[:, 0] < thickness)
                )[0]
            elif wall_pos == "x=L":
                wall_indices = np.where(
                    (self.types == self.FLUID)
                    & (self.positions[:, 0] > self.L - thickness)
                )[0]
            elif wall_pos == "y=0":
                wall_indices = np.where(
                    (self.types == self.FLUID) & (self.positions[:, 1] < thickness)
                )[0]
            elif wall_pos == "y=L":
                wall_indices = np.where(
                    (self.types == self.FLUID)
                    & (self.positions[:, 1] > self.L - thickness)
                )[0]
            else:
                raise ValueError(f"Unknown wall position: {wall_pos}")

            self.types[wall_indices] = self.WALL

        print(f"Created walls at positions: {positions}")

    def set_wall_velocity(self, velocity):
        self.wall_velocity = np.array(velocity)
        print(f"Set wall velocity to {self.wall_velocity}")

    def calculate_temperature(self):
        non_wall_indices = np.where(self.types != self.WALL)[0]
        if len(non_wall_indices) > 0:
            non_wall_velocities = self.velocities[non_wall_indices]
            K = 0.5 * np.sum(non_wall_velocities**2)
            # Temperature = 2K/N in 2D (kB=1 in DPD units)
            self.temperature = K / len(non_wall_indices)
        else:
            self.temperature = 0.0

        return self.temperature

    def calculate_forces(self):
        pairs = self.cell_list.get_potential_pairs()

        dpd_forces, potential_energy = compute_dpd_forces(
            self.positions,
            self.velocities,
            self.types,
            pairs,
            self.L,
            self.rc,
            self.a_matrix,
            self.sigma,
            self.gamma,
            self.dt,
        )

        bond_forces, bond_energy = compute_bond_forces(
            self.positions, self.bonds, self.L, self.K_S, self.r_S
        )

        self.forces = dpd_forces + bond_forces

        if self.body_force != 0:
            self.forces = apply_body_force(
                self.forces, self.types, self.body_force, self.WALL
            )

        self.potential_energy = potential_energy
        self.bond_energy = bond_energy

        return self.forces

    def step(self):
        self.time += self.dt

        wall_indices = np.where(self.types == self.WALL)[0]
        non_wall_indices = np.where(self.types != self.WALL)[0]
        intended_wall_velocities = self.velocities[wall_indices].copy()

        self.velocities[non_wall_indices] += (
            0.5 * self.dt * self.forces[non_wall_indices]
        )
        self.positions[non_wall_indices] += self.dt * self.velocities[non_wall_indices]

        if len(wall_indices) > 0:
            self.positions[wall_indices] += self.dt * intended_wall_velocities

        self.positions = np.mod(self.positions, self.L)

        self.update_cell_list()

        self.calculate_forces()

        self.velocities[non_wall_indices] += (
            0.5 * self.dt * self.forces[non_wall_indices]
        )

        if len(wall_indices) > 0:
            self.velocities[wall_indices] = intended_wall_velocities

        if len(non_wall_indices) > 0:
            self.kinetic_energy = 0.5 * np.sum(self.velocities[non_wall_indices] ** 2)
            self.calculate_temperature()
        else:
            self.kinetic_energy = 0.0
            self.temperature = 0.0

        self.record_measurements()

    def record_measurements(self):
        self.time_history.append(self.time)
        self.temperature_history.append(self.temperature)
        self.kinetic_energy_history.append(self.kinetic_energy)
        self.potential_energy_history.append(self.potential_energy)
        self.total_energy_history.append(
            self.kinetic_energy + self.potential_energy + self.bond_energy
        )

    def reset_measurements(self):
        self.time_history = []
        self.temperature_history = []
        self.kinetic_energy_history = []
        self.potential_energy_history = []
        self.total_energy_history = []
        self.time = 0.0

    def get_velocity_profile(self, direction="y", bins=20, component="x"):
        non_wall_indices = np.where(self.types != self.WALL)[0]

        if direction == "x":
            pos = self.positions[non_wall_indices, 0]
        else:
            pos = self.positions[non_wall_indices, 1]

        if component == "x":
            vel = self.velocities[non_wall_indices, 0]
        else:
            vel = self.velocities[non_wall_indices, 1]

        bin_edges = np.linspace(0, self.L, bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        bin_indices = np.minimum(np.floor(pos / self.L * bins).astype(int), bins - 1)

        velocities = np.zeros(bins)
        counts = np.zeros(bins)

        for i in range(len(pos)):
            bin_idx = bin_indices[i]
            velocities[bin_idx] += vel[i]
            counts[bin_idx] += 1

        valid_bins = counts > 0
        velocities[valid_bins] = velocities[valid_bins] / counts[valid_bins]

        self.velocity_profile_bins = bin_centers
        self.velocity_profile = velocities

        return bin_centers, velocities

    def get_density_profile(self, direction="y", bins=20, particle_type=None):
        if particle_type is not None:
            indices = np.where(self.types == particle_type)[0]
        else:
            indices = np.arange(len(self.positions))

        if direction == "x":
            pos = self.positions[indices, 0]
        else:
            pos = self.positions[indices, 1]

        bin_edges = np.linspace(0, self.L, bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_width = self.L / bins

        bin_indices = np.minimum(np.floor(pos / self.L * bins).astype(int), bins - 1)

        counts = np.zeros(bins)
        for i in range(len(pos)):
            bin_idx = bin_indices[i]
            counts[bin_idx] += 1

        bin_area = bin_width * self.L
        densities = counts / bin_area

        self.density_profile_bins = bin_centers
        self.density_profile = densities

        return bin_centers, densities
