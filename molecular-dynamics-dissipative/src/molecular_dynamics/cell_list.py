"""
Cell list implementation for efficient neighbor search in particle simulations
"""

import numpy as np


class CellList:
    def __init__(self, box_size, cutoff):
        self.L = box_size
        self.rc = cutoff

        self.num_cells = max(1, int(self.L / self.rc))
        self.cell_size = self.L / self.num_cells

        self.reset()

        self._neighbor_offsets = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                self._neighbor_offsets.append((di, dj))

    def reset(self):
        self.cells = [[] for _ in range(self.num_cells * self.num_cells)]
        self.particle_cells = {}

    def cell_index(self, x, y):
        x = np.mod(x, self.L)
        y = np.mod(y, self.L)

        i = int(x / self.cell_size)
        j = int(y / self.cell_size)

        if i == self.num_cells:
            i = self.num_cells - 1
        if j == self.num_cells:
            j = self.num_cells - 1

        # Convert 2D cell coordinates to 1D index
        return i + j * self.num_cells

    def get_neighbor_cells(self, cell_idx):
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
        self.reset()

        for i, pos in enumerate(positions):
            cell_idx = self.cell_index(pos[0], pos[1])
            self.cells[cell_idx].append(i)
            self.particle_cells[i] = cell_idx

    def get_potential_pairs(self):
        pairs = []
        processed_cells = set()

        for cell_idx in range(len(self.cells)):
            particles = self.cells[cell_idx]

            for i, p1 in enumerate(particles):
                for p2 in particles[i + 1 :]:
                    pairs.append((p1, p2))

            neighbor_cells = self.get_neighbor_cells(cell_idx)
            for neighbor_idx in neighbor_cells:
                if neighbor_idx == cell_idx:
                    continue

                cell_pair = tuple(sorted([cell_idx, neighbor_idx]))
                if cell_pair in processed_cells:
                    continue

                processed_cells.add(cell_pair)

                for p1 in particles:
                    for p2 in self.cells[neighbor_idx]:
                        pairs.append((p1, p2))

        return pairs

    def get_neighbors(self, particle_idx, positions):
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
