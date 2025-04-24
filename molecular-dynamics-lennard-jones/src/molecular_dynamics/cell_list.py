"""
Cell list implementation for efficient neighbor search in particle simulations
"""

import numpy as np


class CellList:
    """
    Cell list implementation for efficient spatial partitioning and neighbor searching
    """

    def __init__(self, box_size, cutoff):
        """
        Initialize cell list for a square box.

        Parameters:
        -----------
        box_size : float
            Size of simulation box (L)
        cutoff : float
            Interaction cutoff distance (rc)
        """
        self.L = box_size
        self.rc = cutoff

        # Determine number of cells in each dimension
        # Each cell should have size >= cutoff
        self.num_cells = max(1, int(self.L / self.rc))
        self.cell_size = self.L / self.num_cells

        # Initialize cell data structures
        self.reset()

        # Pre-calculate neighboring cell patterns for efficiency
        self._neighbor_offsets = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                self._neighbor_offsets.append((di, dj))

    def reset(self):
        """Reset cell list data structures"""
        # Map from cell index to list of particle indices in that cell
        self.cells = [[] for _ in range(self.num_cells * self.num_cells)]

        # Map from particle index to containing cell index
        self.particle_cells = {}

    def cell_index(self, x, y):
        """
        Convert spatial coordinates to cell index.

        Parameters:
        -----------
        x, y : float
            Spatial coordinates

        Returns:
        --------
        int : Cell index in 1D representation
        """
        # Ensure coordinates are within [0, L)
        x = np.mod(x, self.L)
        y = np.mod(y, self.L)

        # Calculate cell coordinates
        i = int(x / self.cell_size)
        j = int(y / self.cell_size)

        # Handle edge case
        if i == self.num_cells:
            i = self.num_cells - 1
        if j == self.num_cells:
            j = self.num_cells - 1

        # Convert 2D cell coordinates to 1D index
        return i + j * self.num_cells

    def get_neighbor_cells(self, cell_idx):
        """
        Get list of neighboring cells (including self) with periodic boundaries.

        Parameters:
        -----------
        cell_idx : int
            Index of the cell

        Returns:
        --------
        list : Indices of neighboring cells
        """
        # Convert 1D index to 2D coordinates
        i = cell_idx % self.num_cells
        j = cell_idx // self.num_cells

        neighbors = []

        # Iterate over 3x3 neighborhood with periodic boundaries
        for di, dj in self._neighbor_offsets:
            ni = (i + di) % self.num_cells
            nj = (j + dj) % self.num_cells
            neighbor_idx = ni + nj * self.num_cells
            neighbors.append(neighbor_idx)

        return neighbors

    def update(self, positions):
        """
        Update cell list with new particle positions.

        Parameters:
        -----------
        positions : numpy.ndarray (N, 2)
            Array of particle positions
        """
        self.reset()

        # Assign particles to cells
        for i, pos in enumerate(positions):
            cell_idx = self.cell_index(pos[0], pos[1])
            self.cells[cell_idx].append(i)
            self.particle_cells[i] = cell_idx

    def get_potential_pairs(self):
        """
        Get all potential interacting pairs of particles.

        Returns:
        --------
        list of tuples: (i, j) pairs of particle indices
        """
        pairs = []
        processed_cells = set()

        # Iterate over all cells
        for cell_idx in range(len(self.cells)):
            # Get particles in this cell
            particles = self.cells[cell_idx]

            # Check pairs within the same cell
            for i, p1 in enumerate(particles):
                for p2 in particles[i + 1 :]:
                    pairs.append((p1, p2))

            # Check pairs with neighboring cells
            neighbor_cells = self.get_neighbor_cells(cell_idx)
            for neighbor_idx in neighbor_cells:
                # Skip if it's the same cell (already processed above)
                if neighbor_idx == cell_idx:
                    continue

                # Create a unique key for this cell pair to avoid duplicates
                cell_pair = tuple(sorted([cell_idx, neighbor_idx]))
                if cell_pair in processed_cells:
                    continue

                processed_cells.add(cell_pair)

                # Check pairs between this cell and the neighbor cell
                for p1 in particles:
                    for p2 in self.cells[neighbor_idx]:
                        pairs.append((p1, p2))

        return pairs

    def get_neighbors(self, particle_idx, positions):
        """
        Get indices of particles potentially within cutoff of given particle.

        Parameters:
        -----------
        particle_idx : int
            Index of the particle
        positions : numpy.ndarray (N, 2)
            Array of particle positions

        Returns:
        --------
        list : Indices of potential neighbors
        """
        if particle_idx not in self.particle_cells:
            return []

        cell_idx = self.particle_cells[particle_idx]
        neighbor_cells = self.get_neighbor_cells(cell_idx)

        neighbors = []
        for neighbor_idx in neighbor_cells:
            for p in self.cells[neighbor_idx]:
                if p != particle_idx:
                    neighbors.append(p)

        return neighbors
