import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from heapq import heappush, heappop
import utils
import cv2
import utils


# ============================================================
#  HEURISTIC (Octile Distance)
# ============================================================
def heuristic(r, c):
    dx = abs(r[0] - c[0])
    dy = abs(r[1] - c[1])
    # D = 1, D_diag = sqrt(2)
    return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)

# ============================================================
#  VISUALISATION 
# ============================================================
def display_map(map_grid, path, simplified_path, start, goal):
    # Define colors for the grid
    cmap = ListedColormap(['white', 'blue', 'red'])
    map_display = np.zeros_like(map_grid, dtype=object)

    # Colors
    map_display[map_grid == -1] = 'red'  # Obstacle
    map_display[map_grid == 0] = 'white'   # Free Space

    # Path
    for position in path:
        if map_display[position] in 'white':
            map_display[position] = 'blue'

    # Convert color names to numbers
    color_mapping = {'white': 0, 'blue': 1, 'red': 2}
    map_numeric_display = np.vectorize(color_mapping.get)(map_display)

    # Show map 
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_numeric_display, cmap=cmap)

    # Set tick positions
    x_positions = np.arange(0, 201, 20)
    y_positions = np.arange(0, 201, 20)
    x_labels = [int(utils.cell_to_cm(col)) for col in x_positions]
    y_labels = [int(utils.arena_height_cm - utils.cell_to_cm(row)) for row in y_positions]

    plt.xticks(x_positions, x_labels)
    plt.yticks(y_positions, y_labels)
    ax.set_xlabel('X Dimension (cm)')
    ax.set_ylabel('Y Dimension (cm)')
    
    # Grid lines
    ax.set_xticks(np.arange(-0.5, map_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, map_grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.15)
    plt.title("Pathfinding | Full Path & Simplified Waypoints")

    # Draw robot footprint at each path cell
    robot_h_cells = utils.cm_to_cell(utils.ROBOT_H)
    robot_w_cells = utils.cm_to_cell(utils.ROBOT_W)
    half_h = robot_h_cells / 2
    half_w = robot_w_cells / 2

    for (cx, cy) in path:
        # Note: (cx, cy) is the center (row, col). For plotting, (col, row) is (x, y).
        row_start = cx - half_h
        col_start = cy - half_w

        rect = plt.Rectangle((col_start, row_start),
               robot_w_cells, robot_h_cells, linewidth=1.0, edgecolor='cyan', facecolor='cyan', alpha=0.05)
        ax.add_patch(rect)

    # Start and Goal (note: scatter uses (x, y) = (col, row))
    ax.scatter(start[1], start[0], s=300, c="blue")
    ax.scatter(goal[1], goal[0], s=300, c="green")
    
    # Overlay simplified path waypoints
    if simplified_path:
        # Plotting expects (x, y) where x is horizontal (column), y is vertical (row)
        simplified_cols = [p[1] for p in simplified_path] # Column (X)
        simplified_rows = [p[0] for p in simplified_path] # Row (Y)

        ax.plot(
            simplified_cols,
            simplified_rows,
            'o',                # Marker style: circle
            color='darkviolet', # Color of the dots
            markersize=5,       # Size of the dots
            markeredgecolor='black', # Outline color
            zorder=3            # Ensure dots are on top
        )

    plt.show()

# ============================================================
# display only map
# ============================================================
def display_grid(map_grid, start=None, goal=None):
    cmap = ListedColormap(['white', 'red', 'green', 'blue'])
    map_display = np.zeros_like(map_grid, dtype=object)

    # Colors
    map_display[map_grid == -1] = 'red'  # Obstacle
    map_display[map_grid == 0] = 'white'   # Free Space

    if start is not None:
        map_display[start] = 'blue'
    if goal is not None:
        map_display[goal] = 'green'

    # Convert color names to numbers
    color_mapping = {'white': 0, 'red': 1, 'green': 2, 'blue': 3}
    map_numeric_display = np.vectorize(color_mapping.get)(map_display)

    # Show map
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_numeric_display, cmap=cmap)

    # Set tick positions
    x_positions = np.arange(0, 201, 20)
    y_positions = np.arange(0, 201, 20)

    # Convert cell indices to cm labels
    x_labels = [int(utils.cell_to_cm(col)) for col in x_positions]
    y_labels = [int(utils.arena_height_cm - utils.cell_to_cm(row)) for row in y_positions]

    plt.xticks(x_positions, x_labels)
    plt.yticks(y_positions, y_labels)
    ax.set_xlabel('X Dimension (cm)')
    ax.set_ylabel('Y Dimension (cm)')

    # Grid lines
    ax.set_xticks(np.arange(-0.5, map_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, map_grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.15)

    plt.show()


# ============================================================
#  GRID EXPANSION using cv2 dilation
# ============================================================

def expand_grid_by_robot(grid, robot_size_cells):
    """
    Expands obstacles in the grid by dilating them using cv2.
    This effectively grows obstacles by robot_size_cells/2 in all directions.

    Args:
        grid: occupancy grid (0=free, -1=obstacle)
        robot_size_cells: robot size in cells (will dilate by half this)

    Returns:
        expanded grid with dilated obstacles
    """
    # Convert to binary image (255 = obstacle, 0 = free)
    binary = np.where(grid == -1, 255, 0).astype(np.uint8)

    # Circular kernel
    kernel_size = max(1, int(robot_size_cells / 2)+utils.SAFETY_MARGIN)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1))

    # Dilate obstacles
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Convert back to occupancy grid format
    expanded_grid = np.where(dilated == 255, -1, 0).astype(np.int8)

    return expanded_grid


# ============================================================
#  PATH FINDER (returns path that the robot has to take)
# ============================================================
def expanded_a_star(grid, start, goal):
    """
    Finds the shortest path using A* with 8-connectivity.
    Robot is treated as a point after expanding obstacles by robot size.

    Args:
        grid: occupancy grid from vision (0=free, -1=obstacle)
        start: (row, col) start position in grid coordinates
        goal: (row, col) goal position in grid coordinates

    Returns:
        path: list of (row, col) tuples, or None if no path found
        expanded_grid: the occupancy grid with expanded obstacles
    """
    # Get robot size in cells and expand obstacles
    robot_size_cells = utils.cm_to_cell(max(utils.ROBOT_H, utils.ROBOT_W))
    expanded_grid = expand_grid_by_robot(grid, robot_size_cells)

    came_from = {}
    g_costs = {start: 0}
    explored = set()

    # Priority Queue: (f_cost, g_cost, position)
    open_set = [(heuristic(start, goal), 0, start)]
    current_pos = start

    while open_set:
        _, current_g_cost, current_pos = heappop(open_set)

        # Stop condition
        if current_pos == goal:
            break

        # Optimization: Don't re-explore if we found a better path already
        if current_pos in explored:
            continue

        explored.add(current_pos)

        # 8-connected neighbors (dx, dy)
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal
        ]

        for dr, dc in directions:
            nx, ny = current_pos[0] + dr, current_pos[1] + dc
            neighbor = (nx, ny)

            # Bounds Check
            if not (0 <= nx < expanded_grid.shape[0] and 0 <= ny < expanded_grid.shape[1]):
                continue

            # Simple obstacle check (robot is a point now)
            if expanded_grid[nx, ny] == -1:
                continue

            # Calculate movement cost
            move_cost = np.sqrt(2) if (abs(dr) == 1 and abs(dc) == 1) else 1
            tentative_g_cost = current_g_cost + move_cost

            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                came_from[neighbor] = current_pos
                f_cost = tentative_g_cost + heuristic(neighbor, goal)
                heappush(open_set, (f_cost, tentative_g_cost, neighbor))

    # Reconstruct path
    if current_pos == goal:
        path = []
        while current_pos != start:
            path.append(current_pos)
            current_pos = came_from[current_pos]
        path.append(start)
        path.reverse()
        return path, expanded_grid

    return None, expanded_grid


def expanded_dijkstra(grid, start, goal):
    """
    Pathfinding with expanded obstacles using Dijkstra's algorithm.
    Robot is treated as a point after expanding obstacles by robot size.

    Args:
        grid: occupancy grid from vision (0=free, -1=obstacle)
        start: (row, col) start position in grid coordinates
        goal: (row, col) goal position in grid coordinates

    Returns:
        path: list of (row, col) tuples, or None if no path found
        expanded_grid: the occupancy grid with expanded obstacles
    """
    # Get robot size in cells
    robot_size_cells = utils.cm_to_cell(max(utils.ROBOT_H, utils.ROBOT_W))

    # Expand obstacles by robot size using cv2 dilation
    expanded_grid = expand_grid_by_robot(grid, robot_size_cells)

    # Run Dijkstra (robot as point)
    came_from = {}
    g_costs = {start: 0}
    explored = set()

    # Priority queue: (g_cost, position)
    open_set = [(0, start)]

    current_pos = start

    while open_set:
        current_g_cost, current_pos = heappop(open_set)

        # Stop condition
        if current_pos == goal:
            break

        # Skip if already explored
        if current_pos in explored:
            continue
        explored.add(current_pos)

        # 8-connected neighbors
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal
        ]

        for dr, dc in directions:
            nx, ny = current_pos[0] + dr, current_pos[1] + dc
            neighbor = (nx, ny)

            # Bounds check
            if not (0 <= nx < expanded_grid.shape[0] and 0 <= ny < expanded_grid.shape[1]):
                continue

            # Simple obstacle check (robot is a point now)
            if expanded_grid[nx, ny] == -1:
                continue

            # Movement cost (sqrt(2) for diagonal, 1 for cardinal)
            move_cost = np.sqrt(2) if (abs(dr) == 1 and abs(dc) == 1) else 1
            tentative_g_cost = current_g_cost + move_cost

            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                came_from[neighbor] = current_pos
                heappush(open_set, (tentative_g_cost, neighbor))

    # Reconstruct path
    if current_pos == goal:
        path = []
        while current_pos != start:
            path.append(current_pos)
            current_pos = came_from[current_pos]
        path.append(start)
        path.reverse()
        return path, expanded_grid

    return None, expanded_grid

# ============================================================
#  PATH SIMPLIFICATION into waypoints
# ============================================================

def get_direction(p1, p2):
    """Calculates the normalized direction vector (dx, dy) between two points."""
    dx = p2[0] - p1[0] # Change in Row (Y-axis)
    dy = p2[1] - p1[1] # Change in Column (X-axis)
    return (np.sign(dx), np.sign(dy))

def simplify_path(full_path):
    """Reduces the full cell-by-cell path to a list of corner points."""
    if not full_path or len(full_path) < 2:
        return full_path

    simplified_path = [full_path[0]] 
    current_direction = get_direction(full_path[0], full_path[1]) 

    for i in range(1, len(full_path) - 1):
        p_current = full_path[i]
        p_next = full_path[i+1]
        
        next_direction = get_direction(p_current, p_next)
        
        # Detect Change: If the direction is different, the current point is a corner
        if next_direction != current_direction:
            simplified_path.append(p_current)
            current_direction = next_direction 

    simplified_path.append(full_path[-1]) 
    
    return simplified_path

# ============================================================
#  EXECUTION 
# ============================================================

def find_path(mode, grid, start, goal):
    """
    Main function to find path using specified mode.

    Args:
        mode: 0 or 1 
        grid: occupancy grid (0=free, -1=obstacle)
        start: (row, col) start position
        goal: (row, col) goal position
    """
    if mode == 1:
        path, expanded_grid = expanded_a_star(grid, start, goal)
    elif mode == 0:
        path, expanded_grid = expanded_dijkstra(grid, start, goal)
    else:
        raise ValueError("Invalid mode. Choose 'a_star' or 'dijkstra'.")

    if path:
        simplified_path = simplify_path(path)
        converted_simplified_path = [utils.grid_to_real(p) for p in simplified_path]
        real_waypts = [[float(wp[0]), float(wp[1])] for wp in converted_simplified_path]
        display_map(grid, path, simplified_path, start, goal)
        return real_waypts, expanded_grid
    else:
        simplified_path = None
        print("No path found.")
        return None, expanded_grid

    

