import numpy as np

class Agent:
    id_counter = 0 
    
    def __init__(self, x, y, age=0, max_age=0):
        self.x = x
        self.y = y
        self.max_age = max_age
        self.age = age
        self.hunger = 0 
        self.id = Agent.id_counter
        Agent.id_counter += 1
        
    def move(self, step_std, domain_size):
        angle = np.random.uniform(0, 2 * np.pi)
        dx = np.cos(angle)
        dy = np.sin(angle)

        step_length = np.random.normal(0, step_std)
        step_length = abs(step_length)
        
        self.x += step_length * dx
        self.y += step_length * dy
        
        self.x = self.x % domain_size
        self.y = self.y % domain_size
        
    def get_subgrid_coords(self, subgrid_size):
        i = int(self.x / subgrid_size)
        j = int(self.y / subgrid_size)
        return (i, j)


class Rabbit(Agent):
    def __init__(self, x, y, max_age=100, age=None):
        if age is None:
            age = np.random.randint(1, max_age)
        super().__init__(x, y, age, max_age)
    
    def replicate(self, prob_replicate):
        if np.random.random() < prob_replicate:
            return Rabbit(self.x, self.y, self.max_age, age=0)
        return None
    
    def is_dead(self):
        return self.age >= self.max_age


class Wolf(Agent):
    def __init__(self, x, y, hunger_threshold=50):
        super().__init__(x, y)
        self.hunger_threshold = hunger_threshold
        self.hunger = 0
    
    def is_dead(self):
        return self.hunger >= self.hunger_threshold


