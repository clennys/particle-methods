import numpy as np
from cell_list import CellList
from forces import compute_dpd_forces, compute_bond_forces, apply_body_force


class DPDSimulation:
    FLUID = 0  # F
    WALL = 1  # W
    TYPE_A = 2  # A
    TYPE_B = 3  # B

    TYPE_MAP = {"F": FLUID, "W": WALL, "A": TYPE_A, "B": TYPE_B}

    def __init__(
        self,
        L=15.0,  # Box size
        density=4.0,  # Particle number density
        dt=0.01,  # Time step
        rc=1.0,  # Cutoff radius for DPD
        sigma=3.0,  # Random force coefficient (adjusted to satisfy sigma^2 = 2*gamma*kBT)
        gamma=4.5,  # Dissipative force coefficient
        kBT=1.0,  # System temperature
        a_matrix=None,  # Conservative force coefficients
        K_S=100.0,  # Bond spring constant
        r_S=0.1,  # Equilibrium bond length
        body_force=0.0,  # External body force for Poiseuille flow
    ):

        self.L = L
        self.density = density
        self.dt = dt
        self.rc = rc
        self.sigma = sigma
        self.gamma = gamma
        self.kBT = kBT

        # Verify DPD thermostat relationship (sigma^2 = 2*gamma*kBT)
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
            # Default a_ij = 25 for all pairs
            self.a_matrix = np.full((4, 4), 25.0)
        else:
            self.a_matrix = a_matrix

        self.positions = np.zeros((self.N_total, 2))
        self.velocities = np.zeros((self.N_total, 2))
        self.forces = np.zeros((self.N_total, 2))
        self.types = np.full(
            self.N_total, self.FLUID, dtype=int
        )  # Start with all fluid

        self.bonds = []  # List of (i, j) pairs of bonded particles
        self.molecules = []  # List of sets of particle indices for each molecule

        self.wall_velocity = np.zeros(2)  # Default: stationary walls

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
        """Initialize particles with random positions in the box"""
        self.positions = np.random.uniform(0, self.L, (self.N_total, 2))

        # Initialize velocities to zero as specified in the homework
        self.velocities = np.zeros((self.N_total, 2))

        # Update cell list
        self.update_cell_list()

    def update_cell_list(self):
        """Update the cell list with current particle positions"""
        self.cell_list.update(self.positions)

    def create_chain_molecules(self, num_chains):
        """
        Create chain molecules with pattern A-A-B-B-B-B-B (2 A particles, 5 B particles)

        Parameters:
        -----------
        num_chains : int
            Number of chain molecules to create
        """
        # Clear existing molecules list
        chain_molecules = []

        # Create new bonds list
        new_bonds = []

        # Particle indices to use for chains
        available_indices = np.where(self.types == self.FLUID)[0]

        if len(available_indices) < num_chains * 7:
            raise ValueError(
                f"Not enough fluid particles to create {num_chains} chains of 7 particles each"
            )

        for i in range(num_chains):
            # Select 7 particles for this chain
            chain_indices = available_indices[i * 7 : (i + 1) * 7]

            # Create a new molecule (list of particle indices)
            chain_molecules.append(set(chain_indices))

            # Set particle types (2 A followed by 5 B)
            self.types[chain_indices[0:2]] = self.TYPE_A
            self.types[chain_indices[2:7]] = self.TYPE_B

            # Create bonds between adjacent particles
            for j in range(6):
                new_bonds.append((int(chain_indices[j]), int(chain_indices[j + 1])))

            # Position the chain particles close to each other
            base_pos = np.random.uniform(self.rc, self.L - self.rc, 2)
            for j, idx in enumerate(chain_indices):
                # Position particles in a line with small offsets
                offset = np.array([j * self.r_S * 1.2, 0]) + np.random.uniform(
                    -0.1, 0.1, 2
                )
                new_pos = base_pos + offset
                # Ensure within box boundaries
                self.positions[idx] = np.mod(new_pos, self.L)

        # Add new chains to molecules list
        self.molecules.extend(chain_molecules)

        # Add new bonds
        self.bonds.extend(new_bonds)

        # Update cell list after modifying positions
        self.update_cell_list()

        print(f"Created {num_chains} chain molecules")

    def create_ring_molecules(self, num_rings, ring_size=9):
        """
        Create ring molecules with specified number of A particles

        Parameters:
        -----------
        num_rings : int
            Number of ring molecules to create
        ring_size : int
            Number of particles in each ring (default: 9)
        """
        # Clear existing molecules list
        ring_molecules = []

        # Create new bonds list
        new_bonds = []

        # Particle indices to use for rings
        available_indices = np.where(self.types == self.FLUID)[0]

        if len(available_indices) < num_rings * ring_size:
            raise ValueError(
                f"Not enough fluid particles to create {num_rings} rings of {ring_size} particles each"
            )

        for i in range(num_rings):
            # Select particles for this ring
            ring_indices = available_indices[i * ring_size : (i + 1) * ring_size]

            # Create a new molecule (set of particle indices)
            ring_molecules.append(set(ring_indices))

            # Set particle types (all A for rings)
            self.types[ring_indices] = self.TYPE_A

            # Create bonds between adjacent particles and close the ring
            for j in range(ring_size):
                next_j = (j + 1) % ring_size
                new_bonds.append((int(ring_indices[j]), int(ring_indices[next_j])))

            # Position the ring particles in a circle
            center = np.random.uniform(2 * self.rc, self.L - 2 * self.rc, 2)
            radius = (
                self.r_S * ring_size / (2 * np.pi)
            )  # Approximate radius to achieve desired bond length

            for j, idx in enumerate(ring_indices):
                # Position around a circle with small random offsets
                angle = 2 * np.pi * j / ring_size
                offset = radius * np.array([np.cos(angle), np.sin(angle)])
                offset += np.random.uniform(-0.05, 0.05, 2)  # Small random perturbation
                new_pos = center + offset
                # Ensure within box boundaries
                self.positions[idx] = np.mod(new_pos, self.L)

        # Add new rings to molecules list
        self.molecules.extend(ring_molecules)

        # Add new bonds
        self.bonds.extend(new_bonds)

        # Update cell list after modifying positions
        self.update_cell_list()

        print(f"Created {num_rings} ring molecules with {ring_size} particles each")

    def create_walls(self, thickness=None, positions=None):
        """
        Create walls by converting fluid particles to wall particles in specified regions.

        Parameters:
        -----------
        thickness : float
            Thickness of the wall (default: rc)
        positions : str or list
            'x' for walls at x=0 and x=L
            'y' for walls at y=0 and y=L
            or a list of wall positions like ['x=0', 'x=L']
        """
        if thickness is None:
            thickness = self.rc

        if positions is None:
            positions = ["y=0", "y=L"]  # Default: walls at y=0 and y=L

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

            # Convert to wall particles
            self.types[wall_indices] = self.WALL

        print(f"Created walls at positions: {positions}")

    def set_wall_velocity(self, velocity):
        """
        Set velocity for wall particles.

        Parameters:
        -----------
        velocity : array-like
            Velocity vector [vx, vy] for wall particles
        """
        self.wall_velocity = np.array(velocity)
        print(f"Set wall velocity to {self.wall_velocity}")

    def calculate_temperature(self):
        """Calculate system temperature from kinetic energy"""
        # Sum kinetic energy of non-wall particles only
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
        """Calculate all forces in the system"""
        # Get pairs from cell list
        pairs = self.cell_list.get_potential_pairs()

        # Calculate DPD forces - now passing the time step
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

        # Calculate bond forces
        bond_forces, bond_energy = compute_bond_forces(
            self.positions, self.bonds, self.L, self.K_S, self.r_S
        )

        # Combine forces
        self.forces = dpd_forces + bond_forces

        # Apply body force if specified (for Poiseuille flow)
        if self.body_force != 0:
            self.forces = apply_body_force(
                self.forces, self.types, self.body_force, self.WALL
            )

        # Store energies
        self.potential_energy = potential_energy
        self.bond_energy = bond_energy

        return self.forces

    # def step(self):
    #     """Perform one time step using velocity-Verlet algorithm"""
    #     # Update time
    #     self.time += self.dt
    #
    #     # First half of velocity update
    #     self.velocities += 0.5 * self.dt * self.forces
    #
    #     # Position update
    #     self.positions += self.dt * self.velocities
    #
    #     # Apply periodic boundary conditions
    #     self.positions = np.mod(self.positions, self.L)
    #
    #     # Update wall particles with prescribed motion
    #     wall_indices = np.where(self.types == self.WALL)[0]
    #     if len(wall_indices) > 0:
    #         # Set velocities for wall particles
    #         self.velocities[wall_indices] = self.wall_velocity
    #
    #         # Additional position update for wall particles if they're moving
    #         if np.any(self.wall_velocity != 0):
    #             self.positions[wall_indices] += self.dt * self.wall_velocity
    #             self.positions[wall_indices] = np.mod(
    #                 self.positions[wall_indices], self.L
    #             )
    #
    #     # Update cell list with new positions
    #     self.update_cell_list()
    #
    #     # Calculate new forces
    #     self.calculate_forces()
    #
    #     # Second half of velocity update
    #     self.velocities += 0.5 * self.dt * self.forces
    #
    #     # Reset wall particle velocities (these should have prescribed values)
    #     if len(wall_indices) > 0:
    #         self.velocities[wall_indices] = self.wall_velocity
    #
    #     # Calculate kinetic energy and temperature
    #     non_wall_indices = np.where(self.types != self.WALL)[0]
    #     self.kinetic_energy = 0.5 * np.sum(self.velocities[non_wall_indices] ** 2)
    #     self.calculate_temperature()
    #
    #     # Record measurements
    #     self.record_measurements()
    def step(self):
        # Update time
        self.time += self.dt

        # --- Store intended wall state ---
        wall_indices = np.where(self.types == self.WALL)[0]
        non_wall_indices = np.where(self.types != self.WALL)[0]
        # Store the velocities walls SHOULD have (set during setup)
        intended_wall_velocities = self.velocities[wall_indices].copy()

        # --- Update velocities (Part 1) - ONLY non-wall particles ---
        # Apply forces only to particles that should accelerate
        self.velocities[non_wall_indices] += 0.5 * self.dt * self.forces[non_wall_indices]

        # --- Update positions ---
        # Update non-wall particles based on their dynamically changing velocity
        self.positions[non_wall_indices] += self.dt * self.velocities[non_wall_indices]
        # Update wall particles based on their PRESCRIBED constant velocity
        if len(wall_indices) > 0:
            self.positions[wall_indices] += self.dt * intended_wall_velocities

        # Apply periodic boundary conditions to ALL
        self.positions = np.mod(self.positions, self.L)

        # Update cell list
        self.update_cell_list()

        # --- Calculate Forces ---
        # Based on new positions
        self.calculate_forces() # Note: Forces ON walls are calculated but won't affect their velocity

        # --- Update velocities (Part 2) - ONLY non-wall particles ---
        self.velocities[non_wall_indices] += 0.5 * self.dt * self.forces[non_wall_indices]

        # --- Ensure wall velocities remain fixed ---
        # (Optional, good practice) Reset wall velocities just in case, though they shouldn't have changed
        if len(wall_indices) > 0:
            self.velocities[wall_indices] = intended_wall_velocities

        # --- Calculate Temperature (excluding walls) ---
        if len(non_wall_indices) > 0:
            self.kinetic_energy = 0.5 * np.sum(self.velocities[non_wall_indices]**2)
            self.calculate_temperature()
        else:
            self.kinetic_energy = 0.0
            self.temperature = 0.0

        # Record measurements
        self.record_measurements()

    def record_measurements(self):
        """Record simulation measurements"""
        self.time_history.append(self.time)
        self.temperature_history.append(self.temperature)
        self.kinetic_energy_history.append(self.kinetic_energy)
        self.potential_energy_history.append(self.potential_energy)
        self.total_energy_history.append(
            self.kinetic_energy + self.potential_energy + self.bond_energy
        )

    def reset_measurements(self):
        """Reset measurement histories"""
        self.time_history = []
        self.temperature_history = []
        self.kinetic_energy_history = []
        self.potential_energy_history = []
        self.total_energy_history = []
        self.time = 0.0

    def get_velocity_profile(self, direction="y", bins=20, component="x"):
        """
        Calculate velocity profile along a specified direction.

        Parameters:
        -----------
        direction : str
            Direction along which to calculate profile ('x' or 'y')
        bins : int
            Number of bins
        component : str
            Velocity component to profile ('x' or 'y')

        Returns:
        --------
        bin_centers : numpy.ndarray
            Centers of the bins
        velocities : numpy.ndarray
            Average velocity in each bin
        """
        # Exclude wall particles
        non_wall_indices = np.where(self.types != self.WALL)[0]

        if direction == "x":
            pos = self.positions[non_wall_indices, 0]
        else:
            pos = self.positions[non_wall_indices, 1]

        if component == "x":
            vel = self.velocities[non_wall_indices, 0]
        else:
            vel = self.velocities[non_wall_indices, 1]

        # Create bins
        bin_edges = np.linspace(0, self.L, bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Bin the data
        bin_indices = np.minimum(np.floor(pos / self.L * bins).astype(int), bins - 1)

        # Calculate average velocity in each bin
        velocities = np.zeros(bins)
        counts = np.zeros(bins)

        for i in range(len(pos)):
            bin_idx = bin_indices[i]
            velocities[bin_idx] += vel[i]
            counts[bin_idx] += 1

        # Avoid division by zero for empty bins
        valid_bins = counts > 0
        velocities[valid_bins] = velocities[valid_bins] / counts[valid_bins]

        self.velocity_profile_bins = bin_centers
        self.velocity_profile = velocities

        return bin_centers, velocities

    def get_density_profile(self, direction="y", bins=20, particle_type=None):
        """
        Calculate number density profile along a specified direction.

        Parameters:
        -----------
        direction : str
            Direction along which to calculate profile ('x' or 'y')
        bins : int
            Number of bins
        particle_type : int or None
            If specified, only calculate density for this particle type

        Returns:
        --------
        bin_centers : numpy.ndarray
            Centers of the bins
        densities : numpy.ndarray
            Number density in each bin
        """
        if particle_type is not None:
            indices = np.where(self.types == particle_type)[0]
        else:
            indices = np.arange(len(self.positions))

        if direction == "x":
            pos = self.positions[indices, 0]
        else:
            pos = self.positions[indices, 1]

        # Create bins
        bin_edges = np.linspace(0, self.L, bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_width = self.L / bins

        # Bin the data
        bin_indices = np.minimum(np.floor(pos / self.L * bins).astype(int), bins - 1)

        # Calculate counts in each bin
        counts = np.zeros(bins)
        for i in range(len(pos)):
            bin_idx = bin_indices[i]
            counts[bin_idx] += 1

        # Convert counts to number density
        # In 2D, volume = area = bin_width * box_width
        bin_area = bin_width * self.L
        densities = counts / bin_area

        self.density_profile_bins = bin_centers
        self.density_profile = densities

        return bin_centers, densities
