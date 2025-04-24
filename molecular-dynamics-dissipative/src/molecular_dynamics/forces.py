import numpy as np


def compute_dpd_forces(positions, velocities, types, pairs, box_size, rc, a_matrix, sigma, gamma):
    # N = positions.shape[0]
    forces = np.zeros_like(positions)
    energy = 0.0  # For monitoring, not used in integration
    
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
        
        w_R = 1.0 - r / rc  # Weight function for random force
        w_D = w_R * w_R     # Weight function for dissipative force
        
        # Generate symmetric random number
        xi_ij = rand_matrix[idx]
        
        # Conservative force
        F_C = a_ij * (1.0 - r / rc) * r_hat
        
        # Dissipative force
        F_D = -gamma * w_D * np.dot(r_hat, vij) * r_hat
        
        # Random force with proper scaling for time step
        F_R = sigma * w_R * xi_ij * r_hat
        
        # Total force
        F_ij = F_C + F_D + F_R
        
        forces[i] += F_ij
        forces[j] -= F_ij
        
        # Add to potential energy (just for monitoring)
        energy += 0.5 * a_ij * (1.0 - r / rc)**2
    
    return forces, energy


def compute_bond_forces(positions, bonds, box_size, k_spring, r0):
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
    f_body = np.array([body_force, 0.0])
    
    # Apply to all particles except wall particles
    for i in range(len(forces)):
        if types[i] != wall_type:
            forces[i] += f_body
    
    return forces
