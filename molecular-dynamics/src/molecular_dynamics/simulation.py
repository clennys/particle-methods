"""
Simulation module for Lennard-Jones 2D particles with improved RDF calculation
"""

import numpy as np
from cell_list import CellList


class Simulation:
    """
    Class to handle the simulation of particles interacting via Lennard-Jones potential
    in a 2D periodic box using the velocity-Verlet algorithm.
    """

    def __init__(
        self,
        N=100,
        L=30.0,
        dt=0.01,
        rc=2.5,
        initial_temp=0.5,
        use_thermostat=False,
        tau_factor=0.0025,
    ):
        """
        Initialize the simulation.

        Parameters:
        -----------
        N : int
            Number of particles
        L : float
            Box size (square domain)
        dt : float
            Time step
        rc : float
            Cutoff radius for Lennard-Jones potential
        initial_temp : float
            Initial temperature
        use_thermostat : bool
            Whether to use Berendsen thermostat
        tau_factor : float
            dt/τ factor for Berendsen thermostat
        """
        self.N = N
        self.L = L
        self.dt = dt
        self.rc = rc
        self.rc2 = rc * rc  # Square of cutoff radius
        self.target_temp = initial_temp
        self.use_thermostat = use_thermostat
        self.tau_factor = tau_factor

        # Initialize energy variables
        self.kinetic_energy = 0.0
        self.potential_energy = 0.0

        # Initialize particles on a lattice
        self.setup_particles()

        # Initialize cell list
        self.cell_list = CellList(L, rc)
        self.update_cell_list()

        # Lennard-Jones potential at cutoff (for energy shift)
        self.u_rc = 4.0 * ((1.0 / self.rc) ** 12 - (1.0 / self.rc) ** 6)

        # For energy and temperature tracking
        self.kinetic_energy_history = []
        self.potential_energy_history = []
        self.total_energy_history = []
        self.temperature_history = []
        self.time_history = []
        self.time = 0.0

        # For RDF
        self.rdf = None
        self.rdf_r = None

        # Calculate initial forces and energies
        self.forces = np.zeros((N, 2))
        self.calculate_forces_and_potential()
        self.calculate_kinetic_energy()
        self.record_measurements()

    def setup_particles(self):
        """Set up particles on a square lattice with randomized velocities"""
        """Initialization of 2D MD program"""
        # Initialize sums
        sumv = np.zeros(2)
        sumv2 = 0

        # Determine grid size based on number of particles
        n_side = int(np.ceil(np.sqrt(self.N)))
        spacing = self.L / n_side

        # Create positions array
        self.positions = np.zeros((self.N, 2))
        self.velocities = np.zeros((self.N, 2))
        self.prev_positions = np.zeros((self.N, 2))

        # Place particles on a square lattice
        particle_count = 0
        for i in range(n_side):
            for j in range(n_side):
                if particle_count < self.N:
                    # Add some noise to avoid perfect lattice
                    noise = np.random.uniform(-0.1 * spacing, 0.1 * spacing, 2)
                    pos = (
                        np.array([i * spacing + spacing / 2, j * spacing + spacing / 2])
                        + noise
                    )
                    # Ensure within boundaries
                    self.positions[particle_count] = np.mod(pos, self.L)

                    # Random velocities in 2D
                    self.velocities[particle_count] = np.array(
                        [(np.random.random() - 0.5), (np.random.random() - 0.5)]
                    )

                    # Accumulate for center of mass velocity and kinetic energy
                    sumv += self.velocities[particle_count]
                    sumv2 += np.dot(
                        self.velocities[particle_count], self.velocities[particle_count]
                    )

                    particle_count += 1
        min_allowed_distance = 0.4 * spacing  # Adjust this value as needed
        # Check all pairs of particles
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Calculate distance with periodic boundary conditions
                dr = self.positions[i] - self.positions[j]
                # Apply minimum image convention
                dr = dr - self.L * np.round(dr / self.L)
                dist = np.sqrt(np.sum(dr**2))

                # Assert that particles aren't too close
                assert (
                    dist > min_allowed_distance
                ), f"Particles {i} and {j} are too close: {dist} < {min_allowed_distance}"

        # Calculate center of mass velocity and mean-squared velocity
        sumv = sumv / self.N
        sumv2 = sumv2 / self.N

        # Scale factor for velocities - using factor of 2 for 2D
        fs = np.sqrt(2 * self.target_temp / sumv2)

        # Set desired kinetic energy and velocity center of mass to zero
        for i in range(self.N):
            self.velocities[i] = (self.velocities[i] - sumv) * fs

            # Position at previous time step for Verlet algorithm
            self.prev_positions[i] = self.positions[i] - self.velocities[i] * self.dt

    def update_cell_list(self):
        """Update the cell list with current particle positions"""
        self.cell_list.update(self.positions)

    def calculate_forces_and_potential(self):
        """Calculate forces and potential energy using cell list"""
        # Reset forces and energy
        self.forces = np.zeros((self.N, 2))
        potential_energy = 0.0

        # Get potential pairs from cell list
        pairs = self.cell_list.get_potential_pairs()

        # Loop over potential pairs of particles
        for i, j in pairs:
            # Calculate displacement vector (respecting periodic boundary conditions)
            dr = self.positions[i] - self.positions[j]
            # Apply minimum image convention
            dr = dr - self.L * np.round(dr / self.L)

            # Calculate squared distance
            r2 = np.sum(dr**2)

            # Add safety check to prevent numerical instability
            if r2 < 1e-10:
                r2 = 1e-10  # Avoid division by zero

            # Only calculate force if particles are within cutoff
            if r2 < self.rc2:
                r2i = 1.0 / r2
                r6i = r2i**3

                # Force magnitude * r
                ff = 48.0 * r2i * r6i * (r6i - 0.5)

                # Update forces (vector calculation)
                force_vec = ff * dr
                self.forces[i] += force_vec
                self.forces[j] -= force_vec

                # Update potential energy
                potential_energy += 4.0 * r6i * (r6i - 1.0) - self.u_rc

        # Store the potential energy
        self.potential_energy = potential_energy

        return potential_energy

    def calculate_kinetic_energy(self):
        """Calculate kinetic energy of the system"""
        # K = 0.5 * m * v^2 (m=1 in reduced units)
        self.kinetic_energy = 0.5 * np.sum(self.velocities**2)
        return self.kinetic_energy

    def get_temperature(self):
        """Calculate temperature from kinetic energy"""
        # In reduced units (kB = 1)
        return self.kinetic_energy / self.N

    @property
    def temperature(self):
        """Get current temperature"""
        return self.get_temperature()

    @property
    def total_energy(self):
        """Get total energy"""
        return self.kinetic_energy + self.potential_energy

    @property
    def total_momentum(self):
        """Get total momentum"""
        return np.sum(self.velocities, axis=0)

    def apply_thermostat(self):
        """Apply Berendsen thermostat"""
        if not self.use_thermostat:
            return

        current_temp = self.get_temperature()
        if current_temp < 1e-10:  # Avoid division by zero
            return

        # Scale factor for Berendsen thermostat
        lambda_factor = np.sqrt(
            1.0 + self.tau_factor * (self.target_temp / current_temp - 1.0)
        )
        self.velocities *= lambda_factor

    def step(self):
        """Perform one time step using velocity-Verlet algorithm"""
        # Update time
        self.time += self.dt

        # First half of velocity update
        self.velocities += 0.5 * self.dt * self.forces

        # Position update
        self.positions += self.dt * self.velocities

        # Apply periodic boundary conditions
        self.positions = np.mod(self.positions, self.L)

        # Update cell list with new positions
        self.update_cell_list()

        # Calculate new forces
        self.calculate_forces_and_potential()

        # Second half of velocity update
        self.velocities += 0.5 * self.dt * self.forces

        # Apply thermostat if enabled
        if self.use_thermostat:
            self.apply_thermostat()
        else:
            # Ensure momentum conservation in NVE ensemble
            total_momentum = np.sum(self.velocities, axis=0)
            self.velocities -= total_momentum / self.N

        # Calculate kinetic energy with updated velocities
        self.calculate_kinetic_energy()

        # Check for numerical instability
        if abs(self.total_energy) > 1e10 or np.isnan(self.total_energy):
            raise RuntimeError("Simulation becoming unstable, reduce time step.")

        # Record measurements
        self.record_measurements()

    def record_measurements(self):
        """Record energy and temperature measurements"""
        self.kinetic_energy_history.append(self.kinetic_energy)
        self.potential_energy_history.append(self.potential_energy)
        self.total_energy_history.append(self.total_energy)
        self.temperature_history.append(self.temperature)
        self.time_history.append(self.time)

    def reset_measurements(self):
        """Reset measurement histories"""
        self.kinetic_energy_history = []
        self.potential_energy_history = []
        self.total_energy_history = []
        self.temperature_history = []
        self.time_history = []
        self.time = 0.0

    def calculate_rdf(self, num_bins=50, max_samples=1):
        """
        Calculate radial distribution function g(r)

        Parameters:
        -----------
        num_bins : int
            Number of bins for RDF calculation
        max_samples : int
            Maximum number of samples to collect
        """
        # Maximum distance is half the box length (to avoid double counting)
        r_max = self.L / 2.0
        # Initialize
        bin_size = r_max / num_bins
        g_r = np.zeros(num_bins)
        sample_count = 0

        # Sample (can be called periodically during simulation)
        def sample():
            nonlocal sample_count
            sample_count += 1
            # Loop over all pairs
            for i in range(self.N - 1):
                for j in range(i + 1, self.N):
                    # Calculate displacement with periodic boundary conditions
                    dr = self.positions[i] - self.positions[j]
                    dr = dr - self.L * np.round(dr / self.L)
                    r = np.sqrt(np.sum(dr**2))
                    # Only count pairs within half the box length
                    if r < r_max:
                        # Determine bin index
                        bin_idx = int(r / bin_size)
                        if bin_idx < num_bins:
                            # Count 2 for both i,j and j,i
                            g_r[bin_idx] += 2

        # Collect samples
        for _ in range(max_samples):
            sample()

        # Calculate final g(r)
        # Generate r values at bin centers
        r_values = np.array([bin_size * (i + 0.5) for i in range(num_bins)])

        # Calculate normalization for 2D system
        # In 2D: volume element = 2πr dr, instead of 4πr²dr for 3D
        rho = self.N / (self.L**2)  # Number density

        normalized_g_r = np.zeros(num_bins)
        # Calculate normalization factors (area of circular shell * density)
        for i in range(num_bins):
            r = r_values[i]
            # Volume (area) of this shell: 2π*r*dr for 2D
            v_shell = 2 * np.pi * r * bin_size
            # Ideal gas number in this shell
            n_id = rho * v_shell * self.N
            # Normalize - avoid division by zero
            if n_id > 0 and sample_count > 0:
                normalized_g_r[i] = g_r[i] / (sample_count * n_id)

        self.rdf_r = r_values
        self.rdf = normalized_g_r

        return self.rdf_r, self.rdf
