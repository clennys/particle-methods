from model import ForestFireModel
import numpy as np
from particles import CellState
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def run_simulation():
    # Create model
    model = ForestFireModel(100, 100)
    
    # Initialize environment
    model.initialize_random_terrain()
    model.set_uniform_wind(direction=np.pi/4, strength=0.5)  # Wind from southwest
    model.set_fuel_heterogeneity()
    model.set_moisture_gradient()
    
    # Set up some barriers (e.g., roads or rivers)
    for i in range(30, 70):
        model.grid[i, 50] = CellState.EMPTY.value
    
    # Ignite fire
    model.ignite(20, 20)
    
    # Set up visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Run simulation with animation
    def animate(frame):
        active = model.update()
        ax.clear()
        model.visualize(ax)
        if not active:
            anim.event_source.stop()
        return ax
    
    anim = FuncAnimation(fig, animate, frames=100, interval=100)
    plt.show()

if __name__ == "__main__":
    run_simulation() 
