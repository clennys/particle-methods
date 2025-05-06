import numpy as np


def compute_dpd_forces(
    positions, velocities, types, pairs, box_size, rc, a_matrix, sigma, gamma, dt
):
    forces = np.zeros_like(positions)
    energy = 0.0  # For monitoring, not used in integration

    rand_matrix = np.random.normal(0, 1, len(pairs))

    for idx, (i, j) in enumerate(pairs):
        rij = positions[i] - positions[j]
        rij = rij - box_size * np.round(rij / box_size)

        r = np.linalg.norm(rij)

        if r >= rc:
            continue

        r_hat = rij / r

        type_i = types[i]
        type_j = types[j]
        a_ij = a_matrix[type_i, type_j]

        vij = velocities[i] - velocities[j]

        w_R = 1.0 - r / rc
        w_D = w_R * w_R

        xi_ij = rand_matrix[idx]

        # Conservative force
        F_C = a_ij * (1.0 - r / rc) * r_hat

        # Dissipative force
        F_D = -gamma * w_D * np.dot(r_hat, vij) * r_hat

        # Random force
        F_R = sigma * w_R * xi_ij * r_hat / np.sqrt(dt)

        F_ij = F_C + F_D + F_R
        forces[i] += F_ij
        forces[j] -= F_ij

        energy += 0.5 * a_ij * (1.0 - r / rc) ** 2

    return forces, energy


def compute_bond_forces(positions, bonds, box_size, k_spring, r0):
    forces = np.zeros_like(positions)
    energy = 0.0

    for i, j in bonds:
        rij = positions[i] - positions[j]
        rij = rij - box_size * np.round(rij / box_size)

        r = np.linalg.norm(rij)

        if r == r0:
            continue

        r_hat = rij / r

        # Harmonic spring force
        F_spring = k_spring * (1.0 - r / r0) * r_hat

        forces[i] += F_spring
        forces[j] -= F_spring

        energy += 0.5 * k_spring * (r - r0) ** 2

    return forces, energy


def apply_body_force(forces, types, body_force, wall_type):
    f_body = np.array([body_force, 0.0])

    for i in range(len(forces)):
        if types[i] != wall_type:
            forces[i] += f_body

    return forces
