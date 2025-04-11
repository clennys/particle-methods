"""
Simulation module for Lennard-Jones 2D particles
"""
import numpy as np
from cell_list import CellList

class Simulation:
    """
    Class to handle the simulation of particles interacting via Lennard-Jones potential
    in a 2D periodic box using the velocity-Verlet algorithm.
    """
    def __init__(self, N=100, L=30.0, dt=0.01, rc=2.5, initial_temp=0.5,
                 use_thermostat=False, tau_factor=0.0025):
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
        
        # Initialize particles on a lattice
        self.setup_particles()
        
        # Initialize cell list
        self.cell_list = CellList(L, rc)
        self.update_cell_list()
        
        # Lennard-Jones potential at cutoff (for energy shift)
        self.u_rc = 4.0 * ((1.0/self.rc)**12 - (1.0/self.rc)**6)
        
        # For energy and temperature tracking
        self.kinetic_energy_history = []
        self.potential_energy_history = []
        self.total_energy_history = []
        self.temperature_history = []
        self.time_history = []
        self.time = 0.0
        
        # Calculate initial forces and energies
        self.forces = np.zeros((N, 2))
        self.calculate_forces_and_potential()
        self.calculate_kinetic_energy()
        self.record_measurements()
    
    def setup_particles(self):
        """Set up particles on a square lattice with randomized velocities"""
        # Determine grid size based on number of particles
        n_side = int(np.ceil(np.sqrt(self.N)))
        spacing = self.L / n_side
        
        # Initialize positions on a square lattice
        positions = []
        for i in range(n_side):
            for j in range(n_side):
                if len(positions) < self.N:
                    # Add some noise to avoid perfect lattice
                    noise = np.random.uniform(-0.1 * spacing, 0.1 * spacing, 2)
                    pos = np.array([i * spacing + spacing/2, j * spacing + spacing/2]) + noise
                    # Ensure within boundaries
                    pos = np.mod(pos, self.L)
                    positions.append(pos)
        
        self.positions = np.array(positions)
        
        # Initialize velocities from Maxwell-Boltzmann distribution
        self.velocities = np.random.normal(0, np.sqrt(self.target_temp), (self.N, 2))
        
        # Ensure zero total momentum
        total_momentum = np.sum(self.velocities, axis=0)
        self.velocities -= total_momentum / self.N
        
        # Scale velocities to match target temperature
        self.calculate_kinetic_energy()
        current_temp = self.get_temperature()
        scale_factor = np.sqrt(self.target_temp / current_temp)
        self.velocities *= scale_factor
    
    def update_cell_list(self):
        """Update the cell list with current particle positions"""
        self.cell_list.update(self.positions)
    
    def calculate_forces_and_potential(self):
        """Calculate forces and potential energy using cell list"""
        # Reset forces and potential energy
        self.forces.fill(0.0)
        self.potential_energy = 0.0
        
        # Use cell list to get potential pairs
        pairs = self.cell_list.get_potential_pairs()
        
        for i, j in pairs:
            # Skip if same particle
            if i == j:
                continue
                
            # Get positions with periodic boundary conditions
            rij = self.positions[j] - self.positions[i]
            
            # Apply minimum image convention
            # rij = rij - self.L * np.round(rij / self.L)
            
            # Calculate distance and check cutoff
            r2 = np.sum(rij**2)
            if r2 < self.rc2:
                r = np.sqrt(r2)
                r6 = (1.0 / r)**6
                r12 = r6**2
                
                # Force calculation (48*epsilon is the coefficient from derivative)
                force_mag = 48.0 * (r12 - 0.5 * r6) / r2
                force = force_mag * rij
                
                # Apply force to both particles (action-reaction)
                self.forces[i] += force
                self.forces[j] -= force
                
                # Add to potential energy (counted once per pair)
                # U(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6] - U(rc)
                self.potential_energy += 4.0 * (r12 - r6) - self.u_rc
    
    def calculate_kinetic_energy(self):
        """Calculate kinetic energy of the system"""
        # K = 0.5 * m * v^2 (m=1 in reduced units)
        self.kinetic_energy = 0.5 * np.sum(self.velocities**2)
        return self.kinetic_energy
    
    def get_temperature(self):
        """Calculate temperature from kinetic energy"""
        # T = 2K/(N*d*kB) where d=2 is dimension and kB=1 in reduced units
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
        lambda_factor = np.sqrt(1.0 + self.tau_factor * (self.target_temp / current_temp - 1.0))
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
        self.apply_thermostat()
        
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
    
    def calculate_rdf(self, bins=50, r_max=None):
        """
        Calculate radial distribution function g(r)
        
        Parameters:
        -----------
        bins : int
            Number of bins for RDF calculation
        r_max : float
            Maximum distance for RDF (default: half box size)
        """
        if r_max is None:
            r_max = self.L / 2.0
        
        # Create histogram bins
        self.rdf_bins = np.linspace(0, r_max, bins + 1)
        dr = self.rdf_bins[1] - self.rdf_bins[0]
        self.rdf_r = self.rdf_bins[:-1] + 0.5 * dr
        
        # Initialize histogram
        hist = np.zeros(bins)
        
        # Calculate all pairwise distances
        for i in range(self.N):
            for j in range(i + 1, self.N):
                rij = self.positions[j] - self.positions[i]
                
                # Apply minimum image convention
                rij = rij - self.L * np.round(rij / self.L)
                
                r = np.sqrt(np.sum(rij**2))
                if r < r_max:
                    # Find the bin index
                    bin_idx = int(r / dr)
                    if bin_idx < bins:
                        hist[bin_idx] += 2  # Count each pair twice
        
        # Normalize
        # Volume element in 2D: 2πr dr
        # Number density: rho = N / L^2
        rho = self.N / (self.L**2)
        norm = 2 * np.pi * self.rdf_r * dr * rho * self.N
        
        # Calculate g(r)
        self.rdf = hist / norm
        
        return self.rdf_r, self.rdf

