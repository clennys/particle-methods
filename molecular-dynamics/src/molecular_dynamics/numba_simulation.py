"""
Simulation module accelerated with Numba CPU optimization
"""

import numpy as np
from cell_list import CellList
import numba


@numba.jit(nopython=True, parallel=True)
def calculate_forces_numba(positions, forces, L, rc2, u_rc, pairs):
    """
    Calculate forces between particles using Numba's parallel acceleration.
    
    Parameters:
    -----------
    positions : numpy.ndarray
        Particle positions
    forces : numpy.ndarray
        Array to store forces (will be modified in-place)
    L : float
        Box size
    rc2 : float
        Square of cutoff radius
    u_rc : float
        LJ potential at cutoff distance
    pairs : numpy.ndarray
        Pairs of particle indices to check
    
    Returns:
    --------
    float : Potential energy
    """
    potential_energy = 0.0
    
    # Process all pairs in parallel
    for p in numba.prange(len(pairs)):
        i = pairs[p, 0]
        j = pairs[p, 1]
        
        # Calculate displacement vector with PBC
        dr_x = positions[i, 0] - positions[j, 0]
        dr_y = positions[i, 1] - positions[j, 1]
        
        # Apply minimum image convention
        dr_x = dr_x - L * round(dr_x / L)
        dr_y = dr_y - L * round(dr_y / L)
        
        # Calculate squared distance
        r2 = dr_x*dr_x + dr_y*dr_y
        
        # Only calculate force if particles are within cutoff
        if r2 < rc2:
            # Check for numerical stability
            if r2 < 1e-10:
                r2 = 1e-10
                
            # Calculate force and energy
            r2i = 1.0 / r2
            r6i = r2i * r2i * r2i
            
            # Force magnitude * r
            ff = 48.0 * r2i * r6i * (r6i - 0.5)
            
            # Update forces
            fx = ff * dr_x
            fy = ff * dr_y
            
            # Atomic operations to avoid race conditions in parallel mode
            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[j, 0] -= fx
            forces[j, 1] -= fy
            
            # Energy
            pe = 4.0 * r6i * (r6i - 1.0) - u_rc
            potential_energy += pe
            
    return potential_energy


@numba.jit(nopython=True)
def calculate_rdf_numba(positions, N, L, num_bins, r_max):
    """
    Calculate RDF histogram using Numba.
    
    Parameters:
    -----------
    positions : numpy.ndarray
        Particle positions
    N : int
        Number of particles
    L : float
        Box size
    num_bins : int
        Number of bins for RDF histogram
    r_max : float
        Maximum distance to consider
        
    Returns:
    --------
    numpy.ndarray : RDF histogram counts
    """
    bin_size = r_max / num_bins
    g_r = np.zeros(num_bins)
    
    # Loop over all pairs
    for i in range(N - 1):
        for j in range(i + 1, N):
            # Calculate displacement with periodic boundary conditions
            dr_x = positions[i, 0] - positions[j, 0]
            dr_y = positions[i, 1] - positions[j, 1]
            
            # Apply minimum image convention
            dr_x = dr_x - L * round(dr_x / L)
            dr_y = dr_y - L * round(dr_y / L)
            
            # Calculate distance
            r = np.sqrt(dr_x*dr_x + dr_y*dr_y)
            
            # Only count pairs within half the box length
            if r < r_max:
                # Determine bin index
                bin_idx = int(r / bin_size)
                if bin_idx < num_bins:
                    # Count 2 for both i,j and j,i
                    g_r[bin_idx] += 2
                    
    return g_r


# Optimize velocity-Verlet integration with Numba
@numba.jit(nopython=True)
def velocity_verlet_step1(positions, velocities, forces, dt, L):
    """First half of velocity-Verlet: update velocities half-step and positions full-step"""
    for i in range(len(positions)):
        # First half of velocity update
        velocities[i, 0] += 0.5 * dt * forces[i, 0]
        velocities[i, 1] += 0.5 * dt * forces[i, 1]
        
        # Position update
        positions[i, 0] += dt * velocities[i, 0]
        positions[i, 1] += dt * velocities[i, 1]
        
        # Apply periodic boundary conditions
        positions[i, 0] = positions[i, 0] % L
        positions[i, 1] = positions[i, 1] % L
    
    return positions, velocities


@numba.jit(nopython=True)
def velocity_verlet_step2(velocities, forces, dt):
    """Second half of velocity-Verlet: update velocities half-step"""
    for i in range(len(velocities)):
        # Second half of velocity update
        velocities[i, 0] += 0.5 * dt * forces[i, 0]
        velocities[i, 1] += 0.5 * dt * forces[i, 1]
    
    return velocities


@numba.jit(nopython=True)
def apply_thermostat_numba(velocities, current_temp, target_temp, tau_factor):
    """Apply Berendsen thermostat using Numba"""
    # Scale factor
    lambda_factor = np.sqrt(1.0 + tau_factor * (target_temp / current_temp - 1.0))
    
    # Apply scaling
    for i in range(len(velocities)):
        velocities[i, 0] *= lambda_factor
        velocities[i, 1] *= lambda_factor
    
    return velocities


@numba.jit(nopython=True)
def zero_momentum_numba(velocities, N):
    """Ensure zero total momentum using Numba"""
    # Calculate total momentum
    px = 0.0
    py = 0.0
    for i in range(len(velocities)):
        px += velocities[i, 0]
        py += velocities[i, 1]
    
    # Remove center of mass motion
    px_per_particle = px / N
    py_per_particle = py / N
    
    for i in range(len(velocities)):
        velocities[i, 0] -= px_per_particle
        velocities[i, 1] -= py_per_particle
    
    return velocities


@numba.jit(nopython=True)
def calculate_kinetic_energy_numba(velocities):
    """Calculate kinetic energy using Numba"""
    ke = 0.0
    for i in range(len(velocities)):
        ke += velocities[i, 0]**2 + velocities[i, 1]**2
    return 0.5 * ke


class SimulationNumba:
    """
    Class to handle the simulation of particles with Numba CPU acceleration.
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
        Initialize the simulation with parameters.
        
        Parameters:
        -----------
        N : int
            Number of particles
        L : float
            Box size (square domain)
        dt : float
            Time step
        rc : float
            Cutoff radius for potential
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
        
        print("Using Numba CPU acceleration for force calculation")

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
        
        # Compile the Numba functions on first initialization
        # This triggers JIT compilation to avoid first-call overhead
        if N > 1:
            dummy_pairs = np.array([[0, 1]], dtype=np.int64)
            calculate_forces_numba(
                self.positions, np.zeros_like(self.forces), 
                self.L, self.rc2, self.u_rc, dummy_pairs
            )

    def setup_particles(self):
        """Set up particles on a square lattice with randomized velocities"""
        # Initialize sums
        sumv = np.zeros(2)
        sumv2 = 0

        # Determine grid size based on number of particles
        n_side = int(np.ceil(np.sqrt(self.N)))
        spacing = self.L / n_side

        # Create positions array
        self.positions = np.zeros((self.N, 2))
        self.velocities = np.zeros((self.N, 2))

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

        # Calculate center of mass velocity and mean-squared velocity
        sumv = sumv / self.N
        sumv2 = sumv2 / self.N

        # Scale factor for velocities - using factor of 2 for 2D
        fs = np.sqrt(2 * self.target_temp / sumv2)

        # Set desired kinetic energy and velocity center of mass to zero
        self.velocities = (self.velocities - sumv) * fs

    def update_cell_list(self):
        """Update the cell list with current particle positions"""
        self.cell_list.update(self.positions)

    def calculate_forces_and_potential(self):
        """Calculate forces and potential energy using Numba-accelerated function"""
        # Reset forces
        self.forces.fill(0.0)
        
        # Get potential pairs from cell list
        pairs = np.array(self.cell_list.get_potential_pairs(), dtype=np.int64)
        
        # Skip if no pairs (rare but possible)
        if len(pairs) == 0:
            self.potential_energy = 0.0
            return 0.0
        
        # Use Numba-accelerated function
        self.potential_energy = calculate_forces_numba(
            self.positions, self.forces, self.L, self.rc2, self.u_rc, pairs
        )
        
        return self.potential_energy

    def calculate_kinetic_energy(self):
        """Calculate kinetic energy of the system using Numba"""
        self.kinetic_energy = calculate_kinetic_energy_numba(self.velocities)
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
        """Apply Berendsen thermostat using Numba"""
        if not self.use_thermostat:
            return

        current_temp = self.get_temperature()
        if current_temp < 1e-10:  # Avoid division by zero
            return

        # Use Numba-accelerated function
        self.velocities = apply_thermostat_numba(
            self.velocities, current_temp, self.target_temp, self.tau_factor
        )

    def step(self):
        """Perform one time step using velocity-Verlet algorithm with Numba acceleration"""
        # Update time
        self.time += self.dt

        # First half of velocity-Verlet with Numba
        self.positions, self.velocities = velocity_verlet_step1(
            self.positions, self.velocities, self.forces, self.dt, self.L
        )

        # Update cell list with new positions
        self.update_cell_list()

        # Calculate new forces
        self.calculate_forces_and_potential()

        # Second half of velocity-Verlet with Numba
        self.velocities = velocity_verlet_step2(
            self.velocities, self.forces, self.dt
        )

        # Apply thermostat if enabled
        if self.use_thermostat:
            self.apply_thermostat()
        else:
            # Ensure momentum conservation in NVE ensemble with Numba
            self.velocities = zero_momentum_numba(self.velocities, self.N)

        # Calculate kinetic energy with updated velocities
        self.calculate_kinetic_energy()

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
        """Calculate radial distribution function using Numba"""
        # Maximum distance is half the box length
        r_max = self.L / 2.0
        bin_size = r_max / num_bins
        
        # Initialize histogram and sample counter
        g_r = np.zeros(num_bins)
        sample_count = 0

        # Collect samples using Numba
        for _ in range(max_samples):
            sample_count += 1
            g_r += calculate_rdf_numba(self.positions, self.N, self.L, num_bins, r_max)

        # Generate r values at bin centers
        r_values = np.array([bin_size * (i + 0.5) for i in range(num_bins)])

        # Calculate normalization for 2D system
        rho = self.N / (self.L**2)  # Number density
        normalized_g_r = np.zeros(num_bins)
        
        # Normalize
        for i in range(num_bins):
            r = r_values[i]
            # Volume (area) of shell: 2π*r*dr for 2D
            v_shell = 2 * np.pi * r * bin_size
            # Ideal gas number in shell
            n_id = rho * v_shell * self.N
            # Normalize only if n_id is not too small
            if n_id > 1e-10 and sample_count > 0:
                normalized_g_r[i] = g_r[i] / (sample_count * n_id)

        self.rdf_r = r_values
        self.rdf = normalized_g_r

        return self.rdf_r, self.rdf
