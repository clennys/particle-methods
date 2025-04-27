import numpy as np


def compute_dpd_forces(positions, velocities, types, pairs, box_size, rc, a_matrix, sigma, gamma, dt):
    """
    Compute DPD forces between particles.
    
    Parameters:
    -----------
    positions : numpy.ndarray (N, 2)
        Particle positions
    velocities : numpy.ndarray (N, 2)
        Particle velocities
    types : numpy.ndarray (N)
        Particle types (integers representing F, W, A, B)
    pairs : list of tuples
        List of potential interacting pairs from cell list
    box_size : float
        Size of the simulation box (L)
    rc : float
        Cutoff radius for DPD interactions
    a_matrix : numpy.ndarray
        Matrix of conservative force coefficients a_ij
    sigma : float
        Random force coefficient
    gamma : float
        Dissipative force coefficient
    dt : float
        Time step for proper scaling of random forces
    
    Returns:
    --------
    numpy.ndarray:
        Forces on each particle
    float:
        Total potential energy (for monitoring, not used in DPD integration)
    """
    # N = positions.shape[0]
    forces = np.zeros_like(positions)
    energy = 0.0  # For monitoring, not used in integration
    
    # Pre-compute random numbers for efficiency
    rand_matrix = np.random.normal(0, 1, len(pairs))
    
    for idx, (i, j) in enumerate(pairs):
        # Calculate displacement vector with periodic boundary conditions
        rij = positions[i] - positions[j]
        rij = rij - box_size * np.round(rij / box_size)
        
        # Calculate distance and normalized vector
        r = np.linalg.norm(rij)
        
        # Skip if particles are too far apart
        if r >= rc:
            continue
            
        # Calculate normalized vector
        r_hat = rij / r
        
        # Get particle types and corresponding conservative force coefficient
        type_i = types[i]
        type_j = types[j]
        a_ij = a_matrix[type_i, type_j]
        
        # Calculate relative velocity
        vij = velocities[i] - velocities[j]
        
        # Calculate weight functions
        w_R = 1.0 - r / rc  # Weight function for random force
        w_D = w_R * w_R     # Weight function for dissipative force
        
        # Generate symmetric random number
        xi_ij = rand_matrix[idx]
        
        # Calculate DPD forces
        # 1. Conservative force
        F_C = a_ij * (1.0 - r / rc) * r_hat
        
        # 2. Dissipative force
        F_D = -gamma * w_D * np.dot(r_hat, vij) * r_hat
        
        # 3. Random force with proper scaling for time step
        F_R = sigma * w_R * xi_ij * r_hat * np.sqrt(dt)
        
        # Total force
        F_ij = F_C + F_D + F_R
        
        # Add forces (action-reaction)
        forces[i] += F_ij
        forces[j] -= F_ij
        
        # Add to potential energy (just for monitoring)
        energy += 0.5 * a_ij * (1.0 - r / rc)**2
    
    return forces, energy


def compute_bond_forces(positions, bonds, box_size, k_spring, r0):
    """
    Compute harmonic spring forces between bonded particles.
    
    Parameters:
    -----------
    positions : numpy.ndarray (N, 2)
        Particle positions
    bonds : list of tuples
        List of (i, j) pairs of bonded particles
    box_size : float
        Size of the simulation box (L)
    k_spring : float
        Spring constant (K_S)
    r0 : float
        Equilibrium bond length (r_S)
    
    Returns:
    --------
    numpy.ndarray:
        Bond forces on each particle
    float:
        Bond energy
    """
    # N = positions.shape[0]
    forces = np.zeros_like(positions)
    energy = 0.0
    
    for i, j in bonds:
        # Calculate displacement vector with periodic boundary conditions
        rij = positions[i] - positions[j]
        rij = rij - box_size * np.round(rij / box_size)
        
        r = np.linalg.norm(rij)
        
        # Skip if particles are at exactly equilibrium distance (rare)
        if r == r0:
            continue
            
        # Calculate normalized vector
        r_hat = rij / r
        
        # Calculate harmonic spring force
        F_spring = k_spring * (1.0 - r / r0) * r_hat
        
        # Add forces (action-reaction)
        forces[i] += F_spring
        forces[j] -= F_spring
        
        # Add to bond energy
        energy += 0.5 * k_spring * (r - r0)**2
    
    return forces, energy


def apply_body_force(forces, types, body_force, wall_type):
    """
    Apply a constant body force to non-wall particles.
    
    Parameters:
    -----------
    forces : numpy.ndarray (N, 2)
        Current forces on particles
    types : numpy.ndarray (N)
        Particle types
    body_force : float
        Magnitude of body force in x-direction
    wall_type : int
        Integer code for wall particles
    
    Returns:
    --------
    numpy.ndarray:
        Updated forces
    """
    f_body = np.array([body_force, 0.0])
    
    # Apply to all particles except wall particles
    for i in range(len(forces)):
        if types[i] != wall_type:
            forces[i] += f_body
    
    return forces
