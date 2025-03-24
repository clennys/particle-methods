import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from predator_prey.agents import *
from predator_prey.grid import *


class Simulation:
    def __init__(self, domain_size=10, initial_rabbits=900, initial_wolves=100, 
                 rabbit_max_age=100, wolf_hunger_threshold=50, step_std=0.5, 
                 capture_radius=0.5, eat_prob=0.02, rabbit_repl_prob=0.02,
                 n_subgrids=10):

        self.domain_size = domain_size
        self.step_std = step_std
        self.capture_radius = capture_radius
        self.eat_prob = eat_prob
        self.rabbit_repl_prob = rabbit_repl_prob
        
        self.rabbits = {} 
        self.wolves = {}
        self.grid = Grid(domain_size, n_subgrids)
        
        self.rabbit_counts = []
        self.wolf_counts = []
        self.time_steps = []
        
        for _ in range(initial_rabbits):
            x = np.random.uniform(0, domain_size)
            y = np.random.uniform(0, domain_size)
            rabbit = Rabbit(x, y, rabbit_max_age)
            self.rabbits[rabbit.id] = rabbit
        
        for _ in range(initial_wolves):
            x = np.random.uniform(0, domain_size)
            y = np.random.uniform(0, domain_size)
            wolf = Wolf(x, y, wolf_hunger_threshold)
            self.wolves[wolf.id] = wolf
    
    def update_grid(self):
        self.grid.reset()
        
        for rabbit in self.rabbits.values():
            self.grid.assign_agent_to_subgrid(rabbit, is_rabbit=True)
        
        for wolf in self.wolves.values():
            self.grid.assign_agent_to_subgrid(wolf, is_rabbit=False)
    
    def calculate_distance(self, agent1, agent2):
        dx = min(abs(agent1.x - agent2.x), 
                 abs(agent1.x - agent2.x - self.domain_size), 
                 abs(agent1.x - agent2.x + self.domain_size))
        dy = min(abs(agent1.y - agent2.y), 
                 abs(agent1.y - agent2.y - self.domain_size), 
                 abs(agent1.y - agent2.y + self.domain_size))
        return np.sqrt(dx**2 + dy**2)
    
    def wolf_eat(self):
        rabbits_eaten = set()
        new_wolves = []
        
        for _, wolf in self.wolves.items():
            grid_coords = wolf.get_subgrid_coords(self.grid.subgrid_size)
            nearby_rabbit_ids = self.grid.get_neighboring_rabbits(grid_coords)
            
            for rabbit_id in nearby_rabbit_ids:
                if rabbit_id in rabbits_eaten:
                    continue
                
                rabbit = self.rabbits.get(rabbit_id)
                if not rabbit:
                    continue
                
                distance = self.calculate_distance(wolf, rabbit)
                if distance <= self.capture_radius and np.random.random() < self.eat_prob:
                    rabbits_eaten.add(rabbit_id)
                    wolf.hunger = 0
                    
                    # Try to replicate
                    if np.random.random() < self.eat_prob:  # Same probability as eating
                        new_wolf = Wolf(wolf.x, wolf.y, wolf.hunger_threshold)
                        new_wolves.append(new_wolf)
        
        # Remove eaten rabbits
        for rabbit_id in rabbits_eaten:
            if rabbit_id in self.rabbits:
                del self.rabbits[rabbit_id]
        
        # Add new wolves
        for wolf in new_wolves:
            self.wolves[wolf.id] = wolf
    
    def step(self):
        # Record current population sizes
        self.rabbit_counts.append(len(self.rabbits))
        self.wolf_counts.append(len(self.wolves))
        self.time_steps.append(len(self.time_steps))
        
        # All agents move
        for rabbit in list(self.rabbits.values()):
            rabbit.move(self.step_std, self.domain_size)
            rabbit.age += 1
        
        for wolf in list(self.wolves.values()):
            wolf.move(self.step_std, self.domain_size)
            wolf.hunger += 1

        self.update_grid()
        self.wolf_eat()
        
        new_rabbits = []
        for rabbit in list(self.rabbits.values()):
            baby = rabbit.replicate(self.rabbit_repl_prob)
            if baby:
                new_rabbits.append(baby)
        
        for rabbit in new_rabbits:
            self.rabbits[rabbit.id] = rabbit
        
        for rabbit_id in list(self.rabbits.keys()):
            if self.rabbits[rabbit_id].is_dead():
                del self.rabbits[rabbit_id]
        
        for wolf_id in list(self.wolves.keys()):
            if self.wolves[wolf_id].is_dead():
                del self.wolves[wolf_id]
    
    def run(self, time_steps=1000, plot_interval=None):
        if plot_interval is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter_rabbits = ax.scatter([], [], c='green', alpha=0.5, label='Rabbits')
            scatter_wolves = ax.scatter([], [], c='red', alpha=0.8, label='Wolves')
            ax.set_xlim(0, self.domain_size)
            ax.set_ylim(0, self.domain_size)
            ax.set_title("Wolf-Rabbit Predator-Prey Model")
            ax.legend()
            
            grid_size = self.domain_size / self.grid.n_subgrids
            for i in range(self.grid.n_subgrids + 1):
                ax.axhline(y=i * grid_size, color='gray', linestyle='--', alpha=0.3)
                ax.axvline(x=i * grid_size, color='gray', linestyle='--', alpha=0.3)
            
            def update(frame):
                self.step()
                
                if frame % plot_interval == 0:
                    rabbit_positions = np.array([[r.x, r.y] for r in self.rabbits.values()])
                    wolf_positions = np.array([[w.x, w.y] for w in self.wolves.values()])
                    
                    if len(self.rabbits) > 0:
                        scatter_rabbits.set_offsets(rabbit_positions)
                    else:
                        scatter_rabbits.set_offsets(np.empty((0, 2)))
                    
                    if len(self.wolves) > 0:
                        scatter_wolves.set_offsets(wolf_positions)
                    else:
                        scatter_wolves.set_offsets(np.empty((0, 2)))
                    
                    ax.set_title(f"Time step: {frame}, Rabbits: {len(self.rabbits)}, Wolves: {len(self.wolves)}")
                
                return scatter_rabbits, scatter_wolves
            
            animation = FuncAnimation(fig, update, frames=time_steps, interval=50, blit=True)
            plt.show()
        else:
            start_time = time.time()
            for t in range(time_steps):
                if t % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Time step: {t}, Rabbits: {len(self.rabbits)}, Wolves: {len(self.wolves)}, Elapsed: {elapsed:.2f}s")
                self.step()
    
    def plot_population_dynamics(self, name):
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_steps, self.rabbit_counts, 'g-', label='Rabbits')
        plt.plot(self.time_steps, self.wolf_counts, 'r-', label='Wolves')
        plt.xlabel('Time Steps')
        plt.ylabel('Population')
        plt.title('Population Dynamics of the Wolf-Rabbit Model')
        plt.legend()
        plt.grid(True)
        plt.savefig('population_dynamics_' + name + '.png')
        plt.show()
