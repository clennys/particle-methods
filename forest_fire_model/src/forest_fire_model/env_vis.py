from forest_fire_model.model import ForestFireModel
from forest_fire_model.particles import CellState, FireParticle
import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import time

def parse_arguments():
    """Parse command line arguments for the combined environment and simulation visualization."""
    parser = argparse.ArgumentParser(description='Forest Fire Environment and Simulation Visualization')
    
    # Grid size parameters
    parser.add_argument('--width', type=int, default=100, help='Width of the simulation grid')
    parser.add_argument('--height', type=int, default=100, help='Height of the simulation grid')
    
    # Ignition parameters
    parser.add_argument('--ignite_x', type=int, default=20, help='X coordinate for initial ignition')
    parser.add_argument('--ignite_y', type=int, default=20, help='Y coordinate for initial ignition')
    parser.add_argument('--multi_ignition', action='store_true', 
                        help='Ignite multiple points for better spread')
    
    # Map selection
    parser.add_argument('--map_type', type=str, default='houses', 
                        choices=['houses', 'forest', 'river', 'wui', 'coastal', 'mixed'],
                        help='Type of map layout to generate')
    
    # Wind parameters
    parser.add_argument('--wind_direction', type=float, default=np.pi/4, 
                        help='Wind direction in radians (0=east, pi/2=north)')
    parser.add_argument('--wind_strength', type=float, default=0.5, 
                        help='Wind strength (0-1)')
    parser.add_argument('--variable_wind', action='store_true',
                        help='Use variable wind instead of uniform wind')
    
    # Environment parameters
    parser.add_argument('--fuel_types', type=int, default=3, 
                        help='Number of different fuel type patches')
    parser.add_argument('--base_moisture', type=float, default=0.2, 
                        help='Base moisture level (0-1)')
    parser.add_argument('--terrain_smoothness', type=int, default=5,
                        help='Terrain smoothness (higher = smoother)')
    
    # Performance parameters
    parser.add_argument('--max_particles', type=int, default=300,
                        help='Maximum number of particles allowed (limits computational load)')
    parser.add_argument('--skip_3d_update', type=int, default=3,
                        help='Only update 3D plot every N frames (higher = better performance)')
    parser.add_argument('--particle_display_limit', type=int, default=200,
                        help='Maximum number of particles to display (improves rendering speed)')
    
    # Slow spread parameters with good balance
    parser.add_argument('--spread_rate', type=float, default=0.01,
                        help='Base fire spread rate (0-1, higher = faster spread)')
    parser.add_argument('--ignition_probability', type=float, default=0.03,
                        help='Base probability of cell ignition (0-1)')
    parser.add_argument('--intensity_decay', type=float, default=0.97,
                        help='Intensity decay rate (higher = slower decay, 0.9-0.99)')
    parser.add_argument('--particle_lifetime', type=int, default=15,
                        help='Base lifetime for particles')
    parser.add_argument('--random_strength', type=float, default=0.03,
                        help='Strength of random movement (0-0.2)')
    parser.add_argument('--initial_particles', type=int, default=15,
                        help='Number of initial particles')
    parser.add_argument('--particle_generation_rate', type=float, default=0.03,
                        help='Rate at which new particles are generated')
    parser.add_argument('--burnout_rate', type=float, default=0.03,
                        help='Rate at which burning cells burnout')
    parser.add_argument('--min_intensity', type=float, default=0.2,
                        help='Minimum intensity before a particle dies')
    
    # Simulation parameters
    parser.add_argument('--frames', type=int, default=500, 
                        help='Maximum number of simulation frames')
    parser.add_argument('--interval', type=int, default=40, 
                        help='Animation interval in milliseconds')
    parser.add_argument('--remove_barriers', action='store_true',
                        help='Remove barriers to allow fire to spread freely')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step size for simulation (smaller = slower progression)')
    
    # Output
    parser.add_argument('--save', action='store_true', 
                        help='Save simulation to a file')
    parser.add_argument('--output', type=str, default='forest_fire_animation.mp4',
                        help='Output filename for saved simulation')
    
    return parser.parse_args()

def create_small_city(model, args):
    """Create a small cluster of houses in the center without streets."""
    
    # Define city center coordinates
    city_center_x = args.width // 2
    city_center_y = args.height // 2
    
    # Create houses in a compact cluster
    num_houses = 25  # Total number of houses
    houses_created = 0
    
    print(f"Creating compact city cluster at center ({city_center_x}, {city_center_y})")
    
    # Generate houses in concentric circles for natural clustering
    for ring in range(3):  # 3 rings of houses
        ring_radius = 6 + ring * 6  # Rings at radius 6, 12, 18
        houses_in_ring = 6 + ring * 4  # More houses in outer rings
        
        for i in range(houses_in_ring):
            if houses_created >= num_houses:
                break
                
            # Calculate angle for this house in the ring
            angle = 2 * np.pi * i / houses_in_ring
            
            # Add some randomness to the position
            actual_radius = ring_radius + np.random.uniform(-2, 2)
            angle_offset = np.random.uniform(-0.3, 0.3)
            
            # Calculate house position
            house_x = int(city_center_x + actual_radius * np.cos(angle + angle_offset))
            house_y = int(city_center_y + actual_radius * np.sin(angle + angle_offset))
            
            # Ensure house is within grid bounds
            if not (5 <= house_x < args.width - 5 and 5 <= house_y < args.height - 5):
                continue
            
            # Random house size (2x2 to 4x4)
            house_size = np.random.randint(2, 5)
            
            # Create the house (rectangular area)
            for x in range(max(0, house_x - house_size//2), 
                          min(args.width, house_x + house_size//2 + 1)):
                for y in range(max(0, house_y - house_size//2), 
                              min(args.height, house_y + house_size//2 + 1)):
                    if 0 <= x < args.width and 0 <= y < args.height:
                        model.grid[x, y] = CellState.EMPTY.value
            
            houses_created += 1
    
    # Add a few more scattered houses to fill gaps
    for _ in range(8):  # Additional scattered houses
        if houses_created >= num_houses + 8:
            break
            
        # Random position within city area
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(3, 20 - 3)
        
        house_x = int(city_center_x + radius * np.cos(angle))
        house_y = int(city_center_y + radius * np.sin(angle))
        
        # Check if position is valid and not too close to existing houses
        if not (3 <= house_x < args.width - 3 and 3 <= house_y < args.height - 3):
            continue
            
        # Check if there's already a house nearby (avoid overlap)
        overlap = False
        for check_x in range(house_x - 3, house_x + 4):
            for check_y in range(house_y - 3, house_y + 4):
                if (0 <= check_x < args.width and 0 <= check_y < args.height and 
                    model.grid[check_x, check_y] == CellState.EMPTY.value):
                    overlap = True
                    break
            if overlap:
                break
        
        if not overlap:
            # Small house size for scattered houses
            house_size = np.random.randint(2, 4)
            
            # Create the house
            for x in range(max(0, house_x - house_size//2), 
                          min(args.width, house_x + house_size//2 + 1)):
                for y in range(max(0, house_y - house_size//2), 
                              min(args.height, house_y + house_size//2 + 1)):
                    if 0 <= x < args.width and 0 <= y < args.height:
                        model.grid[x, y] = CellState.EMPTY.value
            
            houses_created += 1
    
    print(f"Created compact city with {houses_created} houses (no streets)")

def create_river_map(model, args):
    """Create a map with a winding river and riparian vegetation."""
    
    # Create a winding river from top to bottom
    river_start_x = args.width // 4
    river_end_x = 3 * args.width // 4
    
    # Generate river path using sine wave with random variations
    for y in range(args.height):
        # Base sine wave for meandering
        progress = y / args.height
        base_x = river_start_x + (river_end_x - river_start_x) * progress
        meander = 15 * np.sin(progress * 4 * np.pi + np.random.uniform(-0.5, 0.5))
        
        river_x = int(base_x + meander)
        river_width = np.random.randint(3, 7)  # Variable river width
        
        # Create river (empty cells)
        for x in range(max(0, river_x - river_width//2), 
                      min(args.width, river_x + river_width//2 + 1)):
            if 0 <= x < args.width:
                model.grid[x, y] = CellState.EMPTY.value
    
    print("Created river map with winding waterway")

def create_urban_map(model, args):
    """Create a Wildland-Urban Interface (WUI) fire map with scattered development and fuel transitions."""
    
    # Create wildland-urban interface with scattered housing clusters
    housing_clusters = [
        {'center': (args.width//6, args.height//3), 'houses': 6, 'defensible_space': 8},
        {'center': (args.width//2, args.height//4), 'houses': 4, 'defensible_space': 6},
        {'center': (2*args.width//3, 3*args.height//5), 'houses': 8, 'defensible_space': 10},
        {'center': (args.width//4, 3*args.height//4), 'houses': 5, 'defensible_space': 7},
        {'center': (4*args.width//5, args.height//5), 'houses': 3, 'defensible_space': 5},
    ]
    
    # Create scattered housing with defensible space
    for cluster in housing_clusters:
        center_x, center_y = cluster['center']
        num_houses = cluster['houses']
        defensible_radius = cluster['defensible_space']
        
        # Place houses within cluster first
        placed_houses = 0
        attempts = 0
        while placed_houses < num_houses and attempts < num_houses * 3:
            # Random position within cluster
            house_x = center_x + np.random.randint(-defensible_radius//2, defensible_radius//2 + 1)
            house_y = center_y + np.random.randint(-defensible_radius//2, defensible_radius//2 + 1)
            
            if 3 <= house_x < args.width - 3 and 3 <= house_y < args.height - 3:
                # Check for overlap with existing houses
                overlap = False
                house_size = np.random.randint(2, 4)
                
                for check_x in range(house_x - house_size, house_x + house_size + 1):
                    for check_y in range(house_y - house_size, house_y + house_size + 1):
                        if (0 <= check_x < args.width and 0 <= check_y < args.height and 
                            model.grid[check_x, check_y] == CellState.EMPTY.value):
                            overlap = True
                            break
                    if overlap:
                        break
                
                if not overlap:
                    # Create house
                    for x in range(max(0, house_x - house_size//2), 
                                  min(args.width, house_x + house_size//2 + 1)):
                        for y in range(max(0, house_y - house_size//2), 
                                      min(args.height, house_y + house_size//2 + 1)):
                            model.grid[x, y] = CellState.EMPTY.value
                    
                    placed_houses += 1
            
            attempts += 1
        
        # NOW create defensible space around cluster (after houses are placed)
        # This ensures moisture modifications happen AFTER base moisture is set
        for x in range(max(0, center_x - defensible_radius), 
                      min(args.width, center_x + defensible_radius + 1)):
            for y in range(max(0, center_y - defensible_radius), 
                          min(args.height, center_y + defensible_radius + 1)):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance <= defensible_radius and model.grid[x, y] == CellState.FUEL.value:
                    # Increase moisture in defensible space (simulates fuel management)
                    # Use max to preserve any existing higher moisture values
                    model.moisture[x, y] = max(model.moisture[x, y], model.moisture[x, y] + 0.3)
    
    # Create additional defensible space immediately around each house
    for i in range(args.width):
        for j in range(args.height):
            if model.grid[i, j] == CellState.EMPTY.value:  # This is a structure
                # Create small defensible space around structure
                for x in range(max(0, i - 3), min(args.width, i + 4)):
                    for y in range(max(0, j - 3), min(args.height, j + 4)):
                        if model.grid[x, y] == CellState.FUEL.value:
                            distance = np.sqrt((x - i)**2 + (y - j)**2)
                            if distance <= 3:
                                # Stronger moisture boost near structures
                                moisture_boost = 0.4 * (1 - distance / 3)
                                model.moisture[x, y] = max(model.moisture[x, y], 
                                                          model.moisture[x, y] + moisture_boost)
    
    # Create a few evacuation routes (not full road grid)
    # Main evacuation route - diagonal across map
    for i in range(args.width):
        route_y = int(args.height * 0.6 + 0.1 * args.height * np.sin(i * 0.1))
        for y in range(max(0, route_y - 1), min(args.height, route_y + 2)):
            if 0 <= y < args.height:
                model.grid[i, y] = CellState.EMPTY.value
    
    # Secondary evacuation route
    for j in range(args.height):
        route_x = int(args.width * 0.3 + 0.05 * args.width * np.cos(j * 0.15))
        for x in range(max(0, route_x - 1), min(args.width, route_x + 2)):
            if 0 <= x < args.width:
                model.grid[x, j] = CellState.EMPTY.value
    
    print("Created WUI fire map with scattered development and defensible space")

def create_mountain_map(model, args):
    """Create a coastal fire map with moisture gradient and variable winds."""
    
    # Create coastline on the left side of the map
    coastline_distance = args.width // 4
    
    # Create irregular coastline
    coastline_positions = []
    for y in range(args.height):
        # Base coastline with some irregularity
        coast_x = coastline_distance + int(5 * np.sin(y * 0.1) + 3 * np.cos(y * 0.05))
        coastline_positions.append(coast_x)
        
        # Create water/beach area
        for x in range(0, min(args.width, coast_x + 3)):
            model.grid[x, y] = CellState.EMPTY.value
    
    # Create moisture gradient - higher near coast, decreasing inland
    if hasattr(model, 'moisture'):
        for x in range(args.width):
            for y in range(args.height):
                if model.grid[x, y] == CellState.FUEL.value:
                    coast_x = coastline_positions[y]
                    distance_from_coast = max(0, x - coast_x)
                    
                    # Moisture decreases with distance from coast
                    moisture_factor = np.exp(-distance_from_coast / 20.0)
                    base_coastal_moisture = 0.6
                    base_inland_moisture = 0.1
                    
                    coastal_moisture = base_coastal_moisture * moisture_factor + base_inland_moisture * (1 - moisture_factor)
                    model.moisture[x, y] = min(1.0, coastal_moisture + np.random.uniform(-0.1, 0.1))
    
    # Create fuel transitions - salt-tolerant near coast, drier inland
    if hasattr(model, 'fuel_types'):
        for x in range(args.width):
            for y in range(args.height):
                if model.grid[x, y] == CellState.FUEL.value:
                    coast_x = coastline_positions[y]
                    distance_from_coast = max(0, x - coast_x)
                    
                    if distance_from_coast < 15:
                        # Coastal scrub - moderate flammability
                        model.fuel_types[x, y] = 0.6 + np.random.uniform(-0.1, 0.1)
                    elif distance_from_coast < 35:
                        # Transition zone - mixed vegetation
                        model.fuel_types[x, y] = 1.0 + np.random.uniform(-0.2, 0.3)
                    else:
                        # Inland - drier, more flammable vegetation
                        model.fuel_types[x, y] = 1.5 + np.random.uniform(-0.2, 0.4)
    
    # Create seasonal creeks and wetlands
    num_creeks = 3
    for creek_id in range(num_creeks):
        # Creek starts inland and flows toward coast
        start_x = args.width - 10 - creek_id * 15
        start_y = np.random.randint(10, args.height - 10)
        
        # Creek meanders toward coast
        current_x, current_y = start_x, start_y
        
        for step in range(50):
            # Add some randomness to creek path
            direction_x = -1 + np.random.uniform(-0.3, 0.3)  # Generally flows toward coast
            direction_y = np.random.uniform(-0.5, 0.5)  # Some meandering
            
            current_x += direction_x
            current_y += direction_y
            
            # Ensure creek stays in bounds
            current_x = max(coastline_distance, min(args.width - 1, current_x))
            current_y = max(0, min(args.height - 1, current_y))
            
            # Create creek bed
            creek_width = np.random.randint(1, 3)
            for x in range(int(current_x) - creek_width, int(current_x) + creek_width + 1):
                for y in range(int(current_y) - creek_width, int(current_y) + creek_width + 1):
                    if 0 <= x < args.width and 0 <= y < args.height:
                        model.grid[x, y] = CellState.EMPTY.value
                        # Increase moisture around creek
                        for mx in range(x-2, x+3):
                            for my in range(y-2, y+3):
                                if 0 <= mx < args.width and 0 <= my < args.height:
                                    dist = np.sqrt((mx-x)**2 + (my-y)**2)
                                    if dist <= 3:
                                        moisture_boost = 0.4 * (1 - dist/3)
                                        model.moisture[mx, my] = min(1.0, model.moisture[mx, my] + moisture_boost)
            
            # Stop if we reach the coast
            if current_x <= coastline_positions[int(current_y)] + 5:
                break
    
    # Create some inland hills/ridges with different vegetation
    num_ridges = 2
    for ridge_id in range(num_ridges):
        ridge_center_x = args.width - 20 - ridge_id * 25
        ridge_center_y = args.height // 2 + (ridge_id - 0.5) * 30
        
        ridge_length = 40
        ridge_width = 15
        
        # Create ridge running north-south
        for y in range(max(0, int(ridge_center_y - ridge_length//2)), 
                      min(args.height, int(ridge_center_y + ridge_length//2))):
            for x in range(max(0, int(ridge_center_x - ridge_width//2)), 
                          min(args.width, int(ridge_center_x + ridge_width//2))):
                distance_from_center = np.sqrt((x - ridge_center_x)**2 + (y - ridge_center_y)**2)
                
                if distance_from_center <= ridge_width//2:
                    # Higher elevation, different fuel type
                    if model.grid[x, y] == CellState.FUEL.value:
                        # Chaparral - more flammable on ridges
                        model.fuel_types[x, y] = 1.8 + np.random.uniform(-0.2, 0.3)
                    
                    # Slightly lower moisture on ridges (dries out faster)
                    if model.grid[x, y] == CellState.FUEL.value:
                        model.moisture[x, y] = max(0.05, model.moisture[x, y] - 0.2)
    
    # Set up variable wind that changes direction (simulating sea/land breeze)
    # This will be handled by the main script, but we can modify the wind field here
    if hasattr(model, 'wind_field'):
        # Create base offshore wind (blowing inland from coast)
        base_wind_strength = 0.4
        for x in range(args.width):
            for y in range(args.height):
                coast_x = coastline_positions[y]
                distance_from_coast = max(0, x - coast_x)
                
                # Wind strength decreases inland
                wind_strength = base_wind_strength * np.exp(-distance_from_coast / 30.0)
                
                # Base wind direction: offshore (toward land)
                wind_direction = 0.0  # East
                
                # Add some turbulence near the coast
                if distance_from_coast < 20:
                    wind_direction += np.random.uniform(-0.3, 0.3)
                    wind_strength *= (1 + np.random.uniform(-0.2, 0.2))
                
                wind_x = wind_strength * np.cos(wind_direction)
                wind_y = wind_strength * np.sin(wind_direction)
                
                model.wind_field[x, y] = [wind_x, wind_y]
    
    print("Created coastal fire map with moisture gradient and natural barriers")

def create_mixed_map(model, args):
    """Create a mixed landscape with multiple features."""
    
    # River in the left third
    for y in range(args.height):
        river_x = args.width // 6 + int(5 * np.sin(y * 0.1))
        for x in range(max(0, river_x - 2), min(args.width, river_x + 3)):
            model.grid[x, y] = CellState.EMPTY.value
    
    # Small town in center-right
    town_x, town_y = 2 * args.width // 3, args.height // 2
    for _ in range(15):
        house_x = town_x + np.random.randint(-15, 16)
        house_y = town_y + np.random.randint(-15, 16)
        
        if 5 <= house_x < args.width - 5 and 5 <= house_y < args.height - 5:
            house_size = np.random.randint(2, 4)
            for x in range(max(0, house_x - house_size//2), 
                          min(args.width, house_x + house_size//2 + 1)):
                for y in range(max(0, house_y - house_size//2), 
                              min(args.height, house_y + house_size//2 + 1)):
                    if 0 <= x < args.width and 0 <= y < args.height:
                        model.grid[x, y] = CellState.EMPTY.value
    
    # Mountain ridge in upper portion
    for x in range(args.width):
        if args.height // 4 <= args.height // 3:
            for y in range(args.height // 4, args.height // 3):
                if np.random.random() < 0.3:
                    model.grid[x, y] = CellState.EMPTY.value
    
    print("Created mixed landscape with river, town, and mountain features")

def create_enhanced_fuel_types(model, args):
    """Create enhanced fuel types with distinct characteristics and colors.
    
    This function preserves existing fuel patterns and only enhances areas
    that are still at default fuel type (1.0).
    """
    
    # Don't reset existing fuel types - preserve map-specific patterns
    # model.fuel_types = np.ones((model.width, model.height))  # REMOVED - was causing overwrites
    
    # Define fuel type characteristics
    fuel_patches = [
        {'type': 2.0, 'color': 'brown', 'name': 'Dry Brush'},      # Very flammable
        {'type': 1.5, 'color': 'darkgreen', 'name': 'Dense Forest'}, # High flammability
        {'type': 1.0, 'color': 'forestgreen', 'name': 'Mixed Forest'}, # Normal
        {'type': 0.7, 'color': 'olive', 'name': 'Light Forest'},    # Medium
        {'type': 0.4, 'color': 'yellowgreen', 'name': 'Grassland'}, # Low flammability
    ]
    
    # Create large patches of each fuel type, but only in areas with default fuel (1.0)
    num_patches_per_type = 3
    
    for fuel_patch in fuel_patches:
        fuel_value = fuel_patch['type']
        
        # Skip if this is the default fuel type (already set)
        if fuel_value == 1.0:
            continue
            
        for _ in range(num_patches_per_type):
            # Random center for patch
            center_x = np.random.randint(10, args.width - 10)
            center_y = np.random.randint(10, args.height - 10)
            
            # Random patch size
            patch_radius = np.random.randint(8, 20)
            
            # Create irregular patch using noise
            for x in range(max(0, center_x - patch_radius), 
                          min(args.width, center_x + patch_radius)):
                for y in range(max(0, center_y - patch_radius), 
                              min(args.height, center_y + patch_radius)):
                    
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    
                    # Only set fuel type if it's fuel (not empty/houses) AND still at default (1.0)
                    if (model.grid[x, y] == CellState.FUEL.value and 
                        distance <= patch_radius and
                        abs(model.fuel_types[x, y] - 1.0) < 0.1):  # Only modify default fuel areas
                        
                        # Add some randomness to patch edges
                        edge_probability = max(0, 1 - (distance / patch_radius) + 
                                             np.random.uniform(-0.3, 0.3))
                        
                        if np.random.random() < edge_probability:
                            model.fuel_types[x, y] = fuel_value
    
    print("Enhanced fuel type distribution while preserving map-specific patterns")
    return fuel_patches

# Monkey-patch the FireParticle class for balanced slow spread
original_init = FireParticle.__init__

def custom_init(self, x, y, intensity=1.0, lifetime=15, spread_rate=0.01, random_strength=0.03, 
                intensity_decay=0.97, min_intensity=0.2):
    original_init(self, x, y, intensity, lifetime)
    self.base_spread_rate = spread_rate
    self.random_strength = random_strength
    self.intensity_decay_base = intensity_decay
    self.min_intensity = min_intensity

original_update = FireParticle.update

def custom_update(self, wind_field, terrain, max_width, max_height, dt=0.1):
    # Age the particle
    self.age += dt
    
    # Update velocity based on wind and terrain
    grid_x, grid_y = int(self.position[0]), int(self.position[1])
    
    # Get wind vector at current position
    if 0 <= grid_x < wind_field.shape[0] and 0 <= grid_y < wind_field.shape[1]:
        wind_vector = wind_field[grid_x, grid_y]
        
        # Apply wind force with some randomness
        random_factor = 1.0 + 0.2 * (np.random.random() - 0.5)
        self.velocity += wind_vector * dt * random_factor
    
    # Terrain effect (uphill acceleration)
    if 0 <= grid_x < terrain.shape[0]-1 and 0 <= grid_y < terrain.shape[1]-1:
        # Calculate gradient (slope) of terrain
        dx = terrain[min(grid_x+1, terrain.shape[0]-1), grid_y] - terrain[grid_x, grid_y]
        dy = terrain[grid_x, min(grid_y+1, terrain.shape[1]-1)] - terrain[grid_x, grid_y]
        
        # Fire spreads faster uphill (positive gradient)
        terrain_effect = np.array([dx, dy]) 
        
        # Stronger effect for steeper slopes
        slope_magnitude = np.sqrt(dx**2 + dy**2)
        slope_factor = 1.0 + 2.0 * slope_magnitude
        
        self.velocity += terrain_effect * 0.05 * dt * slope_factor
    
    # Ensure minimal baseline movement even without wind or terrain
    if np.linalg.norm(self.velocity) < self.base_spread_rate:
        # Generate random direction for base spread
        random_direction = np.random.rand(2) * 2 - 1
        random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-8)
        
        # Apply base spread rate in random direction
        self.velocity += random_direction * self.base_spread_rate * dt * 1.2
    
    # Apply random movement with customizable strength
    random_direction = np.random.rand(2) * 2 - 1
    random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-8)
    random_strength = self.random_strength * self.intensity
    self.velocity += random_direction * random_strength * dt
    
    # Limit maximum velocity but ensure minimum velocity
    max_speed = 0.8 * self.intensity
    min_speed = self.base_spread_rate * 0.3
    
    current_speed = np.linalg.norm(self.velocity)
    if current_speed > max_speed:
        self.velocity = (self.velocity / current_speed) * max_speed
    elif current_speed < min_speed and current_speed > 0:
        self.velocity = (self.velocity / current_speed) * min_speed
    
    # Update position
    self.position += self.velocity * dt
    
    # Store path for visualization
    self.path.append((self.position[0], self.position[1]))
    if len(self.path) > self.max_path_length:
        self.path.pop(0)
    
    # Check boundaries
    if not self.is_inbounds(max_width, max_height):
        self.intensity = 0.0
    
    # Balanced intensity decay
    age_factor = min(1.0, self.age / self.lifetime)
    decay_rate = self.intensity_decay_base - (0.05 * age_factor)
    self.intensity *= decay_rate

# Override is_active to use customizable minimum intensity
def custom_is_active(self):
    """Check if the particle is still active using custom minimum intensity"""
    return self.age < self.lifetime and self.intensity > self.min_intensity

# Monkey-patch the ForestFireModel.update method for reliable slow spread
original_model_update = ForestFireModel.update

def custom_model_update(self, dt=0.1):
    """Update the simulation by one time step with balanced slow spread
    
    Args:
        dt (float): Time step size
        
    Returns:
        bool: Whether fire is still active
    """
    new_grid = self.grid.copy()
    new_particles = []
    
    # Update existing particles
    for particle in self.particles:
        particle.update(self.wind_field, self.terrain, self.width, self.height, dt)
        
        if particle.is_active():
            new_particles.append(particle)
            
            # Check for ignition of nearby cells
            x, y = int(particle.position[0]), int(particle.position[1])
            
            # Update max fire spread distance
            if self.ignition_point:
                dist = np.sqrt((x - self.ignition_point[0])**2 + (y - self.ignition_point[1])**2)
                self.fire_spread_distance = max(self.fire_spread_distance, dist)
            
            # Balanced ignition radius
            ignition_radius = int(particle.intensity * 1.8)
            
            for i in range(max(0, x-ignition_radius), min(self.width, x+ignition_radius+1)):
                for j in range(max(0, y-ignition_radius), min(self.height, y+ignition_radius+1)):
                    # Distance from particle to cell
                    distance = np.sqrt((i-particle.position[0])**2 + (j-particle.position[1])**2)
                    
                    if distance <= ignition_radius:
                        # Cell is close enough to potentially ignite
                        if self.grid[i, j] == CellState.FUEL.value:
                            # Balanced ignition probability
                            ignition_prob = (
                                (particle.intensity * 1.0) *
                                (1 - distance/ignition_radius) *
                                self.fuel_types[i, j] *
                                (1 - self.moisture[i, j]) *
                                0.7
                            )
                            
                            # Add base probability
                            ignition_prob = max(self.base_ignition_probability, min(0.7, ignition_prob))
                            
                            # Ignite based on probability
                            if np.random.random() < ignition_prob:
                                new_grid[i, j] = CellState.BURNING.value
                                
                                # Create new particles at a balanced rate
                                if np.random.random() < 0.4:
                                    intensity = np.random.uniform(0.7, 1.0) * particle.intensity
                                    lifetime = np.random.randint(10, 15)
                                    new_particles.append(FireParticle(i, j, intensity, lifetime))
    
    # Update cell states
    for i in range(self.width):
        for j in range(self.height):
            if self.grid[i, j] == CellState.BURNING.value:
                # Each burning cell has a chance to burn out
                if np.random.random() < self.burnout_rate:
                    new_grid[i, j] = CellState.BURNED.value
                    
                # Generate new particles at a balanced rate
                if np.random.random() < self.particle_generation_rate:
                    intensity = np.random.uniform(0.7, 1.0)
                    lifetime = np.random.randint(10, 15)
                    new_particles.append(FireParticle(i, j, intensity, lifetime))
    
    self.grid = new_grid
    self.particles = new_particles
    
    # Check if fire has completely died out
    active = len(new_particles) > 0 or np.sum(self.grid == CellState.BURNING.value) > 0
    
    return active

def run_combined_visualization():
    """Combined environment and fire simulation visualization."""
    args = parse_arguments()
    
    # Apply custom init to FireParticle
    def init_override(self, x, y, intensity=1.0, lifetime=args.particle_lifetime):
        custom_init(self, x, y, intensity, lifetime, 
                   spread_rate=args.spread_rate,
                   random_strength=args.random_strength,
                   intensity_decay=args.intensity_decay,
                   min_intensity=args.min_intensity)
    
    FireParticle.__init__ = init_override
    FireParticle.is_active = custom_is_active
    
    # Apply custom update to ForestFireModel
    ForestFireModel.update = custom_model_update
    
    # Create model with specified dimensions
    model = ForestFireModel(args.width, args.height)
    
    # Adjust model parameters for balanced spread
    model.base_ignition_probability = args.ignition_probability
    model.particle_generation_rate = args.particle_generation_rate
    model.initial_particles = args.initial_particles
    model.burnout_rate = args.burnout_rate
    
    # Initialize environment
    model.initialize_random_terrain(smoothness=args.terrain_smoothness)
    
    # Set base moisture and fuel types BEFORE map creation
    model.set_moisture_gradient(base_moisture=args.base_moisture)
    model.fuel_types = np.ones((model.width, model.height))  # Initialize base fuel types
    
    # Set wind field before map creation (maps may modify if needed)
    if args.variable_wind:
        model.set_variable_wind(base_direction=args.wind_direction, 
                               base_strength=args.wind_strength,
                               variability=0.2)
    else:
        model.set_uniform_wind(direction=args.wind_direction, 
                              strength=args.wind_strength)
    
    # Create map layout based on selected type (will modify moisture/fuel as needed)
    if not args.remove_barriers:
        if args.map_type == 'houses':
            create_small_city(model, args)
        elif args.map_type == 'river':
            create_river_map(model, args)
        elif args.map_type == 'wui':
            create_urban_map(model, args)  # Now creates WUI map with proper moisture modification
        elif args.map_type == 'coastal':
            create_mountain_map(model, args)  # Now creates coastal map with proper fuel transitions
        elif args.map_type == 'mixed':
            create_mixed_map(model, args)
        elif args.map_type == 'forest':
            print("Created pure forest map (no barriers)")
        
        print(f"Map type: {args.map_type}")
    
    # Apply enhanced fuel types only for forest maps or when explicitly requested
    if args.map_type == 'forest' or (args.fuel_types > 5 and args.map_type not in ['coastal', 'wui']):
        fuel_patches = create_enhanced_fuel_types(model, args)
    else:
        # Create a simple fuel patches list for legend consistency
        fuel_patches = [
            {'type': 2.0, 'color': 'brown', 'name': 'Dry Brush'},
            {'type': 1.5, 'color': 'darkgreen', 'name': 'Dense Forest'},
            {'type': 1.0, 'color': 'forestgreen', 'name': 'Mixed Forest'},
            {'type': 0.7, 'color': 'olive', 'name': 'Light Forest'},
            {'type': 0.4, 'color': 'yellowgreen', 'name': 'Grassland'},
        ]
    
    # Create custom visualization function with fuel type colors
    def visualize_with_fuel_colors(model, ax=None, show_particles=True):
        """Enhanced visualization with fuel type colors."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create fuel type background
        fuel_display = np.zeros((model.width, model.height, 3))  # RGB array
        
        # Define colors for each fuel type
        fuel_color_map = {
            0.4: [0.6, 0.8, 0.2],    # Grassland - yellow-green
            0.7: [0.4, 0.6, 0.2],    # Light Forest - olive
            1.0: [0.1, 0.4, 0.1],    # Mixed Forest - forest green
            1.5: [0.0, 0.3, 0.0],    # Dense Forest - dark green
            2.0: [0.5, 0.3, 0.1],    # Dry Brush - brown
        }
        
        # Set fuel colors
        for i in range(model.width):
            for j in range(model.height):
                if model.grid[i, j] == CellState.FUEL.value:
                    fuel_value = model.fuel_types[i, j]
                    # Find closest fuel type
                    closest_fuel = min(fuel_color_map.keys(), 
                                     key=lambda x: abs(x - fuel_value))
                    fuel_display[i, j] = fuel_color_map[closest_fuel]
                elif model.grid[i, j] == CellState.BURNING.value:
                    fuel_display[i, j] = [1.0, 0.0, 0.0]  # Red
                elif model.grid[i, j] == CellState.BURNED.value:
                    fuel_display[i, j] = [0.0, 0.0, 0.0]  # Black
                elif model.grid[i, j] == CellState.EMPTY.value:
                    fuel_display[i, j] = [0.7, 0.7, 0.9]  # Light blue
        
        # Display the fuel map
        ax.imshow(fuel_display.transpose(1, 0, 2), origin='lower', 
                 extent=[0, model.width, 0, model.height])
        
        # Add particles only if requested
        if show_particles and model.particles:
            particle_x = [p.position[0] for p in model.particles]
            particle_y = [p.position[1] for p in model.particles]
            intensity = [p.intensity for p in model.particles]
            
            sizes = [i * 30 for i in intensity]
            ax.scatter(particle_x, particle_y, s=sizes, color='yellow', alpha=0.7)
        
        # Add terrain contours only for right plot
        if show_particles:
            terrain_contour = ax.contour(
                np.arange(model.width), 
                np.arange(model.height), 
                model.terrain.T, 
                levels=8, 
                colors='white', 
                alpha=0.3, 
                linewidths=0.8
            )
        
        # Add statistics
        burned_percent = np.sum(model.grid == CellState.BURNED.value) / (model.width * model.height) * 100
        burning_count = np.sum(model.grid == CellState.BURNING.value)
        fuel_left = np.sum(model.grid == CellState.FUEL.value)
        
        info_text = (
            f"Fire spread: {model.fire_spread_distance:.1f} units\n"
            f"Burned: {burned_percent:.1f}%\n"
            f"Active fires: {burning_count}\n"
            f"Fuel remaining: {fuel_left}"
        )
        
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        ax.set_title(f"Fire Simulation with Particles - {args.map_type.title()}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        return ax
    
    # Create figure with 2 subplots side by side
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # 3D terrain view (left)
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Fire simulation (right)
    ax_right = fig.add_subplot(gs[0, 1])
    
    # Create a meshgrid for 3D plotting
    x = np.arange(0, args.width, 1)
    y = np.arange(0, args.height, 1)
    X, Y = np.meshgrid(x, y)
    
    # Plot the terrain as a 3D surface
    terrain_surf = ax_3d.plot_surface(X, Y, model.terrain.T, 
                                   cmap='terrain', alpha=0.7,
                                   linewidth=0, antialiased=True)
    
    # Add colorbar for the terrain
    fig.colorbar(terrain_surf, ax=ax_3d, shrink=0.5, aspect=5, label='Elevation')
    
    # Set labels for 3D plot
    ax_3d.set_title(f"3D Terrain - {args.map_type.title()} Map")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Elevation")
    
    # Create legend for fuel types and fire states (for right plot)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.5, 0.3, 0.1], label='Dry Brush (Very Flammable)'),
        Patch(facecolor=[0.0, 0.3, 0.0], label='Dense Forest (High Flammable)'),
        Patch(facecolor=[0.1, 0.4, 0.1], label='Mixed Forest (Normal)'),
        Patch(facecolor=[0.4, 0.6, 0.2], label='Light Forest (Medium)'),
        Patch(facecolor=[0.6, 0.8, 0.2], label='Grassland (Low Flammable)'),
        Patch(facecolor=[1.0, 0.0, 0.0], label='Burning'),
        Patch(facecolor=[0.0, 0.0, 0.0], label='Burned'),
        Patch(facecolor=[0.7, 0.7, 0.9], label='Empty (Buildings/Water/Rock)'),
    ]
    
    # Ignite fire
    if args.multi_ignition:
        # Ignite multiple points for better spread
        ignite_points = [
            (args.ignite_x, args.ignite_y),
            (args.ignite_x + 10, args.ignite_y),
            (args.ignite_x - 10, args.ignite_y),
            (args.ignite_x, args.ignite_y + 10),
            (args.ignite_x, args.ignite_y - 10)
        ]
        for x, y in ignite_points:
            if 0 <= x < args.width and 0 <= y < args.height:
                model.ignite(x, y)
    else:
        model.ignite(args.ignite_x, args.ignite_y)
    
    # Initialize variables for animation
    fire_points_3d = None
    frame_count = 0
    last_frame_time = time.time()
    fps_history = []
    
    # Function to limit number of particles for performance
    def limit_particles(model, max_particles):
        if len(model.particles) > max_particles:
            # Sort particles by intensity (keep the most intense ones)
            model.particles.sort(key=lambda p: p.intensity, reverse=True)
            model.particles = model.particles[:max_particles]
            return True
        return False
    
    # Run simulation with animation
    def animate(frame):
        nonlocal fire_points_3d, frame_count, last_frame_time, fps_history
        
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - last_frame_time
        fps = 1.0 / max(elapsed, 0.001)  # Avoid division by zero
        fps_history.append(fps)
        if len(fps_history) > 20:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        last_frame_time = current_time
        
        # Limit particles for performance
        particles_limited = limit_particles(model, args.max_particles)
        
        # Update the model with custom time step size
        active = model.update(dt=args.dt)
        frame_count += 1
        
        # Clear right axis and redraw with fuel types and particles
        ax_right.clear()
        visualize_with_fuel_colors(model, ax_right, show_particles=True)
        
        # Add legend to right plot
        ax_right.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), 
                       fontsize=8, framealpha=0.9)
        
        # Add FPS counter and parameters to right plot
        performance_text = (
            f"FPS: {avg_fps:.1f}\n"
            f"Particles: {len(model.particles)}\n"
            f"Map: {args.map_type.title()}\n"
            f"Spread Rate: {args.spread_rate}\n"
            f"Ignition Prob: {args.ignition_probability}\n"
            f"Burned: {np.sum(model.grid == CellState.BURNED.value)} cells"
        )
        ax_right.text(0.02, 0.98, performance_text, transform=ax_right.transAxes, fontsize=9,
                      verticalalignment='top', horizontalalignment='left',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Update 3D visualization periodically - show realistic moving particles
        if frame_count % args.skip_3d_update == 0:
            # Remove old fire points if they exist
            if fire_points_3d:
                fire_points_3d.remove()
                fire_points_3d = None
            
            # Update 3D plot with fire particles (realistic moving visualization)
            if model.particles:
                # Get subset of particles to display
                display_particles = model.particles[:min(len(model.particles), args.particle_display_limit)]
                
                # Get particle positions and intensities
                particle_x = [p.position[0] for p in display_particles]
                particle_y = [p.position[1] for p in display_particles]
                particle_z = [model.terrain[int(min(p.position[0], model.terrain.shape[0]-1)), 
                                           int(min(p.position[1], model.terrain.shape[1]-1))] + 0.1 
                             for p in display_particles]
                intensity = [p.intensity for p in display_particles]
                
                # Use intensity for point size and color
                sizes = [i * 30 for i in intensity]
                colors = [(1.0, min(1.0, i*0.5 + 0.5), 0.0) for i in intensity]
                
                # Create realistic fire particles visualization
                fire_points_3d = ax_3d.scatter(particle_x, particle_y, particle_z, 
                                             c=colors, s=sizes, marker='o', alpha=0.7)
            
            # Add burned areas as black dots (less frequently for performance)
            burned_indices = np.where(model.grid == CellState.BURNED.value)
            if burned_indices[0].size > 0 and frame_count % (args.skip_3d_update * 2) == 0:
                # Sample burned areas for performance
                sample_size = min(1000, burned_indices[0].size)
                if burned_indices[0].size > sample_size:
                    sample_indices = np.random.choice(burned_indices[0].size, sample_size, replace=False)
                    burned_x = burned_indices[0][sample_indices]
                    burned_y = burned_indices[1][sample_indices]
                else:
                    burned_x = burned_indices[0]
                    burned_y = burned_indices[1]
                
                burned_z = [model.terrain[x, y] + 0.02 for x, y in zip(burned_x, burned_y)]
                
                ax_3d.scatter(burned_x, burned_y, burned_z, 
                            c='black', s=8, marker='s', alpha=0.6, label='Burned')
            
            # Update title with fire information
            ax_3d.set_title(f"3D {args.map_type.title()} - Time: {frame_count}, Active Fires: {len(model.particles)}")
        
        # Stop animation if fire is no longer active or max frames reached
        if not active or frame_count >= args.frames:
            print(f"Simulation ended at time step {frame_count}")
            print(f"Final state: {np.sum(model.grid == CellState.FUEL.value)} unburned cells, {np.sum(model.grid == CellState.BURNED.value)} burned cells")
            anim.event_source.stop()
        
        return ax_3d, ax_right
    
    plt.tight_layout()
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=args.frames, interval=args.interval, blit=False)
    
    # Save animation if requested
    if args.save:
        try:
            from matplotlib.animation import FFMpegWriter
            print(f"Saving animation to {args.output}...")
            writer = FFMpegWriter(fps=15, metadata=dict(artist='ForestFireModel'), bitrate=1800)
            anim.save(args.output, writer=writer)
            print(f"Animation saved successfully to {args.output}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Trying alternative writer...")
            try:
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=10)
                gif_output = args.output.replace('.mp4', '.gif')
                anim.save(gif_output, writer=writer)
                print(f"Animation saved as GIF to {gif_output}")
            except Exception as e2:
                print(f"Error saving as GIF: {e2}")
                print("Animation will only be displayed, not saved.")
    
    plt.show()

if __name__ == "__main__":
    run_combined_visualization()
