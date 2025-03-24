from collections import defaultdict
class Grid:
    def __init__(self, domain_size, n_subgrids):
        self.domain_size = domain_size
        self.n_subgrids = n_subgrids
        self.subgrid_size = domain_size / n_subgrids
        self.reset()
    
    def reset(self):
        self.rabbit_grid = defaultdict(set)
        self.wolf_grid = defaultdict(set)
    
    def assign_agent_to_subgrid(self, agent, is_rabbit):
        grid_coords = agent.get_subgrid_coords(self.subgrid_size)
        if is_rabbit:
            self.rabbit_grid[grid_coords].add(agent.id)
        else:
            self.wolf_grid[grid_coords].add(agent.id)
    
    def get_neighboring_coords(self, grid_coords):
        i, j = grid_coords
        neighbors = []
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni = (i + di) % self.n_subgrids
                nj = (j + dj) % self.n_subgrids
                neighbors.append((ni, nj))
        
        return neighbors
    
    def get_neighboring_rabbits(self, grid_coords):
        neighbor_coords = self.get_neighboring_coords(grid_coords)
        rabbit_ids = set()
        
        for coords in neighbor_coords:
            rabbit_ids.update(self.rabbit_grid[coords])
        
        return rabbit_ids
