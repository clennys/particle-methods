from forest_fire_model.particles import CellState
import numpy as np


def create_small_city(model, args):
    """Create a small cluster of houses in the center without streets."""

    # Define city center coordinates
    city_center_x = args.width // 2
    city_center_y = args.height // 2

    # Create houses in a compact cluster
    num_houses = 25
    houses_created = 0

    print(f"Creating compact city cluster at center ({city_center_x}, {city_center_y})")

    for ring in range(3):  # 3 rings of houses
        ring_radius = 6 + ring * 6  # Rings at radius 6, 12, 18
        houses_in_ring = 6 + ring * 4  # More houses in outer rings

        for i in range(houses_in_ring):
            if houses_created >= num_houses:
                break

            angle = 2 * np.pi * i / houses_in_ring

            actual_radius = ring_radius + np.random.uniform(-2, 2)
            angle_offset = np.random.uniform(-0.3, 0.3)

            house_x = int(city_center_x + actual_radius * np.cos(angle + angle_offset))
            house_y = int(city_center_y + actual_radius * np.sin(angle + angle_offset))

            if not (5 <= house_x < args.width - 5 and 5 <= house_y < args.height - 5):
                continue

            # Random house size (2x2 to 4x4)
            house_size = np.random.randint(2, 5)

            for x in range(
                max(0, house_x - house_size // 2),
                min(args.width, house_x + house_size // 2 + 1),
            ):
                for y in range(
                    max(0, house_y - house_size // 2),
                    min(args.height, house_y + house_size // 2 + 1),
                ):
                    if 0 <= x < args.width and 0 <= y < args.height:
                        model.grid[x, y] = CellState.EMPTY.value

            houses_created += 1

    for _ in range(8):
        if houses_created >= num_houses + 8:
            break

        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(3, 20 - 3)

        house_x = int(city_center_x + radius * np.cos(angle))
        house_y = int(city_center_y + radius * np.sin(angle))

        if not (3 <= house_x < args.width - 3 and 3 <= house_y < args.height - 3):
            continue

        overlap = False
        for check_x in range(house_x - 3, house_x + 4):
            for check_y in range(house_y - 3, house_y + 4):
                if (
                    0 <= check_x < args.width
                    and 0 <= check_y < args.height
                    and model.grid[check_x, check_y] == CellState.EMPTY.value
                ):
                    overlap = True
                    break
            if overlap:
                break

        if not overlap:
            house_size = np.random.randint(2, 4)

            for x in range(
                max(0, house_x - house_size // 2),
                min(args.width, house_x + house_size // 2 + 1),
            ):
                for y in range(
                    max(0, house_y - house_size // 2),
                    min(args.height, house_y + house_size // 2 + 1),
                ):
                    if 0 <= x < args.width and 0 <= y < args.height:
                        model.grid[x, y] = CellState.EMPTY.value

            houses_created += 1

    print(f"Created compact city with {houses_created} houses (no streets)")


def create_river_map(model, args):
    """Create a map with a winding river and riparian vegetation."""

    river_start_x = args.width // 4
    river_end_x = 3 * args.width // 4

    for y in range(args.height):
        progress = y / args.height
        base_x = river_start_x + (river_end_x - river_start_x) * progress
        meander = 15 * np.sin(progress * 4 * np.pi + np.random.uniform(-0.5, 0.5))

        river_x = int(base_x + meander)
        river_width = np.random.randint(3, 7)  # Variable river width

        # Create river (empty cells)
        for x in range(
            max(0, river_x - river_width // 2),
            min(args.width, river_x + river_width // 2 + 1),
        ):
            if 0 <= x < args.width:
                model.grid[x, y] = CellState.EMPTY.value

    print("Created river map with winding waterway")


def create_urban_map(model, args):
    """Create a Wildland-Urban Interface (WUI) fire map with scattered development and fuel transitions."""

    housing_clusters = [
        {
            "center": (args.width // 6, args.height // 3),
            "houses": 6,
            "defensible_space": 8,
        },
        {
            "center": (args.width // 2, args.height // 4),
            "houses": 4,
            "defensible_space": 6,
        },
        {
            "center": (2 * args.width // 3, 3 * args.height // 5),
            "houses": 8,
            "defensible_space": 10,
        },
        {
            "center": (args.width // 4, 3 * args.height // 4),
            "houses": 5,
            "defensible_space": 7,
        },
        {
            "center": (4 * args.width // 5, args.height // 5),
            "houses": 3,
            "defensible_space": 5,
        },
    ]

    for cluster in housing_clusters:
        center_x, center_y = cluster["center"]
        num_houses = cluster["houses"]
        defensible_radius = cluster["defensible_space"]

        placed_houses = 0
        attempts = 0
        while placed_houses < num_houses and attempts < num_houses * 3:
            house_x = center_x + np.random.randint(
                -defensible_radius // 2, defensible_radius // 2 + 1
            )
            house_y = center_y + np.random.randint(
                -defensible_radius // 2, defensible_radius // 2 + 1
            )

            if 3 <= house_x < args.width - 3 and 3 <= house_y < args.height - 3:
                overlap = False
                house_size = np.random.randint(2, 4)

                for check_x in range(house_x - house_size, house_x + house_size + 1):
                    for check_y in range(
                        house_y - house_size, house_y + house_size + 1
                    ):
                        if (
                            0 <= check_x < args.width
                            and 0 <= check_y < args.height
                            and model.grid[check_x, check_y] == CellState.EMPTY.value
                        ):
                            overlap = True
                            break
                    if overlap:
                        break

                if not overlap:
                    for x in range(
                        max(0, house_x - house_size // 2),
                        min(args.width, house_x + house_size // 2 + 1),
                    ):
                        for y in range(
                            max(0, house_y - house_size // 2),
                            min(args.height, house_y + house_size // 2 + 1),
                        ):
                            model.grid[x, y] = CellState.EMPTY.value

                    placed_houses += 1

            attempts += 1

        for x in range(
            max(0, center_x - defensible_radius),
            min(args.width, center_x + defensible_radius + 1),
        ):
            for y in range(
                max(0, center_y - defensible_radius),
                min(args.height, center_y + defensible_radius + 1),
            ):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if (
                    distance <= defensible_radius
                    and model.grid[x, y] == CellState.FUEL.value
                ):
                    model.moisture[x, y] = max(
                        model.moisture[x, y], model.moisture[x, y] + 0.3
                    )

    for i in range(args.width):
        for j in range(args.height):
            if model.grid[i, j] == CellState.EMPTY.value:  # This is a structure
                for x in range(max(0, i - 3), min(args.width, i + 4)):
                    for y in range(max(0, j - 3), min(args.height, j + 4)):
                        if model.grid[x, y] == CellState.FUEL.value:
                            distance = np.sqrt((x - i) ** 2 + (y - j) ** 2)
                            if distance <= 3:
                                moisture_boost = 0.4 * (1 - distance / 3)
                                model.moisture[x, y] = max(
                                    model.moisture[x, y],
                                    model.moisture[x, y] + moisture_boost,
                                )

    for i in range(args.width):
        route_y = int(args.height * 0.6 + 0.1 * args.height * np.sin(i * 0.1))
        for y in range(max(0, route_y - 1), min(args.height, route_y + 2)):
            if 0 <= y < args.height:
                model.grid[i, y] = CellState.EMPTY.value

    for j in range(args.height):
        route_x = int(args.width * 0.3 + 0.05 * args.width * np.cos(j * 0.15))
        for x in range(max(0, route_x - 1), min(args.width, route_x + 2)):
            if 0 <= x < args.width:
                model.grid[x, j] = CellState.EMPTY.value

    print("Created WUI fire map with scattered development and defensible space")


def create_coastal_map(model, args):
    """Create a coastal fire map with moisture gradient and variable winds."""

    coastline_distance = args.width // 4

    coastline_positions = []
    for y in range(args.height):
        coast_x = coastline_distance + int(5 * np.sin(y * 0.1) + 3 * np.cos(y * 0.05))
        coastline_positions.append(coast_x)

        for x in range(0, min(args.width, coast_x + 3)):
            model.grid[x, y] = CellState.EMPTY.value

    if hasattr(model, "moisture"):
        for x in range(args.width):
            for y in range(args.height):
                if model.grid[x, y] == CellState.FUEL.value:
                    coast_x = coastline_positions[y]
                    distance_from_coast = max(0, x - coast_x)

                    moisture_factor = np.exp(-distance_from_coast / 20.0)
                    base_coastal_moisture = 0.4
                    base_inland_moisture = 0.1

                    coastal_moisture = (
                        base_coastal_moisture * moisture_factor
                        + base_inland_moisture * (1 - moisture_factor)
                    )
                    model.moisture[x, y] = min(
                        1.0, coastal_moisture + np.random.uniform(-0.1, 0.1)
                    )

    if hasattr(model, "fuel_types"):
        for x in range(args.width):
            for y in range(args.height):
                if model.grid[x, y] == CellState.FUEL.value:
                    coast_x = coastline_positions[y]
                    distance_from_coast = max(0, x - coast_x)

                    if distance_from_coast < 15:
                        model.fuel_types[x, y] = 0.6 + np.random.uniform(-0.1, 0.1)
                    elif distance_from_coast < 35:
                        model.fuel_types[x, y] = 1.0 + np.random.uniform(-0.2, 0.3)
                    else:
                        model.fuel_types[x, y] = 1.5 + np.random.uniform(-0.2, 0.4)

    num_creeks = 3
    for creek_id in range(num_creeks):
        start_x = args.width - 10 - creek_id * 15
        start_y = np.random.randint(10, args.height - 10)

        current_x, current_y = start_x, start_y

        for step in range(50):
            direction_x = -1 + np.random.uniform(-0.3, 0.3)
            direction_y = np.random.uniform(-0.5, 0.5)

            current_x += direction_x
            current_y += direction_y

            current_x = max(coastline_distance, min(args.width - 1, current_x))
            current_y = max(0, min(args.height - 1, current_y))

            creek_width = np.random.randint(1, 3)
            for x in range(
                int(current_x) - creek_width, int(current_x) + creek_width + 1
            ):
                for y in range(
                    int(current_y) - creek_width, int(current_y) + creek_width + 1
                ):
                    if 0 <= x < args.width and 0 <= y < args.height:
                        model.grid[x, y] = CellState.EMPTY.value
                        for mx in range(x - 2, x + 3):
                            for my in range(y - 2, y + 3):
                                if 0 <= mx < args.width and 0 <= my < args.height:
                                    dist = np.sqrt((mx - x) ** 2 + (my - y) ** 2)
                                    if dist <= 3:
                                        moisture_boost = 0.4 * (1 - dist / 3)
                                        model.moisture[mx, my] = min(
                                            1.0, model.moisture[mx, my] + moisture_boost
                                        )

            if current_x <= coastline_positions[int(current_y)] + 5:
                break

    num_ridges = 2
    for ridge_id in range(num_ridges):
        ridge_center_x = args.width - 20 - ridge_id * 25
        ridge_center_y = args.height // 2 + (ridge_id - 0.5) * 30

        ridge_length = 40
        ridge_width = 15

        for y in range(
            max(0, int(ridge_center_y - ridge_length // 2)),
            min(args.height, int(ridge_center_y + ridge_length // 2)),
        ):
            for x in range(
                max(0, int(ridge_center_x - ridge_width // 2)),
                min(args.width, int(ridge_center_x + ridge_width // 2)),
            ):
                distance_from_center = np.sqrt(
                    (x - ridge_center_x) ** 2 + (y - ridge_center_y) ** 2
                )

                if distance_from_center <= ridge_width // 2:
                    # Higher elevation, different fuel type
                    if model.grid[x, y] == CellState.FUEL.value:
                        # Chaparral - more flammable on ridges
                        model.fuel_types[x, y] = 1.8 + np.random.uniform(-0.2, 0.3)

                    # Slightly lower moisture on ridges (dries out faster)
                    if model.grid[x, y] == CellState.FUEL.value:
                        model.moisture[x, y] = max(0.05, model.moisture[x, y] - 0.2)

    if hasattr(model, "wind_field"):
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
                    wind_strength *= 1 + np.random.uniform(-0.2, 0.2)

                wind_x = wind_strength * np.cos(wind_direction)
                wind_y = wind_strength * np.sin(wind_direction)

                model.wind_field[x, y] = [wind_x, wind_y]

    print("Created coastal fire map with moisture gradient and natural barriers")


def create_mixed_map(model, args):
    """Create a mixed landscape with multiple features."""

    for y in range(args.height):
        river_x = args.width // 6 + int(5 * np.sin(y * 0.1))
        for x in range(max(0, river_x - 2), min(args.width, river_x + 3)):
            model.grid[x, y] = CellState.EMPTY.value

    town_x, town_y = 2 * args.width // 3, args.height // 2
    for _ in range(15):
        house_x = town_x + np.random.randint(-15, 16)
        house_y = town_y + np.random.randint(-15, 16)

        if 5 <= house_x < args.width - 5 and 5 <= house_y < args.height - 5:
            house_size = np.random.randint(2, 4)
            for x in range(
                max(0, house_x - house_size // 2),
                min(args.width, house_x + house_size // 2 + 1),
            ):
                for y in range(
                    max(0, house_y - house_size // 2),
                    min(args.height, house_y + house_size // 2 + 1),
                ):
                    if 0 <= x < args.width and 0 <= y < args.height:
                        model.grid[x, y] = CellState.EMPTY.value

    for x in range(args.width):
        if args.height // 4 <= args.height // 3:
            for y in range(args.height // 4, args.height // 3):
                if np.random.random() < 0.3:
                    model.grid[x, y] = CellState.EMPTY.value

    print("Created mixed landscape with river, town, and mountain features")
