import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.path import Path # Used for robust grid creation
from heapq import heappush, heappop

# ============================================================
#  CONSTANTS
# ============================================================
# Robot footprint (width × height), centered on the path cell
robot_w = 22
robot_h = 22

# Grid resolution is 200x200 (1m / 0.005m)
GRID_DIM = 200
ARENA_SIZE_CM = 100
CELL_SIZE_CM = ARENA_SIZE_CM / GRID_DIM  


# ============================================================
#  AXIS CONVERSION FUNCTIONS
# ============================================================

def cell_to_cm(cell_index):
    """Converts cell index (0-200) to distance in cm (0-100)."""
    return cell_index * CELL_SIZE_CM

def cm_to_cell(cm_value):
    """Converts distance in cm (0-100) to cell index (0-200)."""
    return cm_value / CELL_SIZE_CM

def real_to_grid(coord):
    return (coord[1]/CELL_SIZE_CM, GRID_DIM-coord[0]/CELL_SIZE_CM)

def grid_to_real(coord):
    return (coord[1]*CELL_SIZE_CM, ARENA_SIZE_CM-coord[0]*CELL_SIZE_CM)

# ============================================================
#  HEURISTIC (Octile Distance)
# ============================================================
def heuristic(a, b):
    # a and b are (row, col)
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    # D = 1, D_diag = sqrt(2)
    return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)


# ============================================================
#  ROBOT COLLISION CHECK (center-based)
# ============================================================
def is_robot_valid(map_grid, cx, cy, robot_h, robot_w):
    """
    Checks whether a robot of size robot_h × robot_w centered at (cx, cy)
    fits entirely inside free space.
    """
    # cx is row index (y), cy is column index (x)
    half_h = robot_h // 2
    half_w = robot_w // 2

    row_start = cx - half_h
    # Adjusted row_end for odd/even size: rows in range [row_start, row_end] must be robot_h cells total
    # e.g., for h=3 (half_h=1), row_start=cx-1, row_end=cx+1 -> 3 cells total.
    row_end   = cx + (robot_h - half_h) 
    
    col_start = cy - half_w
    col_end   = cy + (robot_w - half_w)

    # Boundary check (row_end and col_end are exclusive indices for slicing)
    if row_start < 0 or col_start < 0 or row_end > map_grid.shape[0] or col_end > map_grid.shape[1]:
        return False

    # Obstacle check: check if any cell in the robot's footprint is occupied (-1)
    # Note: map_grid[row_start:row_end, col_start:col_end]
    footprint = map_grid[row_start:row_end, col_start:col_end]
    
    # Check if any cell in the footprint is an obstacle (-1)
    if np.any(footprint == -1):
        return False

    return True


# ============================================================
#  VISUALISATION 
# ============================================================
def display_map(map_grid, path, simplified_path, start, goal, explored):
    # Define colors for the grid
    cmap = ListedColormap(['white', 'black', 'blue', 'green', 'red'])
    map_display = np.zeros_like(map_grid, dtype=object)

    # Colors
    map_display[map_grid == -1] = 'red'  # Obstacle
    map_display[map_grid == 0] = 'white'   # Free Space

    # Explored cells (only mark if it was free space)
    """ for position in explored:
        if map_display[position] == 'white':
            map_display[position] = 'grey' """

    # Path
    for position in path:
        if map_display[position] in 'white':
            map_display[position] = 'blue'

    

    # Convert color names to numbers
    color_mapping = {'white': 0, 'black': 1, 'blue': 2,
                     'green': 3, 'red': 4, 'grey': 5}
    map_numeric_display = np.vectorize(color_mapping.get)(map_display)

    # Show map 
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_numeric_display, cmap=cmap)

    # Set tick positions
    x_labels, y_labels = grid_to_real((np.arange(0, 201, 20), np.arange(0, 201, 20)))

    plt.xticks(np.arange(0, 201, 20), x_labels)
    plt.yticks(np.arange(0, 201, 20), y_labels)
    ax.set_xlabel('X Dimension (cm)')
    ax.set_ylabel('Y Dimension (cm)')
    
    # Grid lines
    ax.set_xticks(np.arange(-0.5, map_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, map_grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.15)
    plt.title("A* Pathfinding | Full Path & Simplified Waypoints")

    # Draw robot footprint at each path cell
    half_h = robot_h / 2
    half_w = robot_w / 2

    for (cx, cy) in path:
        # Note: (cx, cy) is the center (row, col). For plotting, (col, row) is (x, y).
        row_start = cx - half_h
        col_start = cy - half_w

        rect = plt.Rectangle((col_start, row_start), # Bottom-left corner for plotting (x, y) = (col_start, row_start)
               robot_w,robot_h, linewidth=1.0,edgecolor='cyan',facecolor='cyan',alpha=0.05)
        ax.add_patch(rect)

    # Start and Goal    
    ax.scatter(SearchStart[0], SearchStart[1], s=300, c="blue")
    ax.scatter(SearchGoal[0], SearchGoal[1], s=300, c="green")
    
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
def display_grid(map_grid):
    cmap = ListedColormap(['white', 'black', 'red', 'green', 'blue'])
    map_display = np.zeros_like(map_grid, dtype=object)

    # Colors
    map_display[map_grid == -1] = 'red'  # Obstacle
    map_display[map_grid == 0] = 'white'   # Free Space

    map_display[SearchStart] = 'blue'
    map_display[SearchGoal] = 'green'

    # Convert color names to numbers
    color_mapping = {'white': 0, 'black': 1, 'red': 2, 'blue': 3,
                     'green': 4}
    map_numeric_display = np.vectorize(color_mapping.get)(map_display)

    # Show map 
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_numeric_display, cmap=cmap)

    # Set tick positions (in pixel indices)
    x_positions = np.arange(0, 201, 20)  
    y_positions = np.arange(0, 201, 20)

    # Convert cell index → centimeters (1 cell = 2 cm)
    x_labels, y_labels = x_labels, y_labels = grid_to_real((np.arange(0, 201, 20), np.arange(0, 201, 20)))

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
#  A* SEARCH
# ============================================================
def grid_search(map_grid, S, G):
    """Finds the shortest path using A* with 8-connectivity and robot collision checking."""
    
    came_from = {}
    g_costs = {S: 0}
    explored = set()

    # Priority Queue: (f_cost, g_cost, position)
    open_set = [(heuristic(S, G), 0, S)] 

    while open_set:
        current_f_cost, current_g_cost, current_pos = heappop(open_set)
        
        # Stop condition
        if current_pos == G:
            break
        
        # Optimization: Don't re-explore if we found a better path already
        if current_pos in explored:
            continue
            
        explored.add(current_pos)

        # 8-connected neighbors (dx, dy)
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1), # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1) # Diagonal
        ]

        for dr, dc in directions:
            nx, ny = current_pos[0] + dr, current_pos[1] + dc
            neighbor = (nx, ny)

            # Bounds Check
            if not (0 <= nx < map_grid.shape[0] and 0 <= ny < map_grid.shape[1]):
                continue

            # Robot Collision Check
            if not is_robot_valid(map_grid, nx, ny, robot_h, robot_w):
                continue

            # Calculate movement cost
            move_cost = np.sqrt(2) if (abs(dr) == 1 and abs(dc) == 1) else 1
            
            # Penalize moving over non-free cells (map_grid[nx, ny] is >= 0)
            tentative_g_cost = current_g_cost + move_cost + map_grid[nx, ny]

            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                came_from[neighbor] = current_pos
                f_cost = tentative_g_cost + heuristic(neighbor, G)
                heappush(open_set, (f_cost, tentative_g_cost, neighbor))

    # Reconstruct path
    if current_pos == G:
        path = []
        while current_pos != S:
            path.append(current_pos)
            current_pos = came_from[current_pos]
        path.append(S)
        path.reverse()
        return path, explored

    return None, explored

# ============================================================
#  OCCUPANCY GRID CREATION
# ============================================================

def create_occupancy_grid(obstacles, grid_dim=GRID_DIM, cell_size_m=CELL_SIZE_CM/100, arena_size_m=ARENA_SIZE_CM/100):
    """
    Creates an occupancy grid (0=free, -1=obstacle) using matplotlib.path.Path.
    """
    grid = np.zeros((grid_dim, grid_dim), dtype=np.int8) # Use -1 for obstacles

    # Create a meshgrid of all cell centers in meters
    x_coords = np.linspace(0.5 * cell_size_m, arena_size_m - 0.5 * cell_size_m, grid_dim)
    y_coords = np.linspace(0.5 * cell_size_m, arena_size_m - 0.5 * cell_size_m, grid_dim)
    
    # Array of all (x, y) points corresponding to cell centers
    X, Y = np.meshgrid(x_coords, y_coords)
    points = np.vstack((X.flatten(), Y.flatten())).T

    occupied_indices = np.zeros(grid_dim * grid_dim, dtype=bool)

    for polygon in obstacles:
        # Reshape polygon vertices to (N, 2)
        verts = polygon.reshape(-1, 2)
        
        # Create a Path object from the polygon vertices
        poly_path = Path(verts)
        
        # Check which cell centers are contained within the polygon
        contained = poly_path.contains_points(points, radius=0)
        
        # Combine the results for all polygons (logical OR)
        occupied_indices = occupied_indices | contained

    # Map Occupancy back to the 2D Grid
    # True -> -1 (Obstacle), False -> 0 (Free)
    grid_flat = occupied_indices.astype(np.int8) * -1
    occupancy_grid = grid_flat.reshape(grid_dim, grid_dim)

    occupancy_grid = np.flipud(occupancy_grid)
    
    return occupancy_grid

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
#  EXECUTION BLOCK
# ============================================================

# Define example obstacles in meters (mimicking cv2.approxPolyDP output)
# 1. Large Triangle (Vertices in meters)
poly1_m_rand = np.array([
    [[0.40, 0.20]],
    [[0.60, 0.45]],
    [[0.35, 0.70]]
], dtype=np.float32)

# 2. Inverted L-Shape (Vertices in meters)
poly2_m_rand = np.array([
    [[0.75, 0.65]],
    [[0.90, 0.65]],
    [[0.90, 0.90]],
    [[0.65, 0.90]],
    [[0.65, 0.80]],
    [[0.75, 0.80]]
], dtype=np.float32)

# 3. Narrow Rectangle (Vertices in meters)
poly3_m_rand = np.array([
    [[0.10, 0.85]],
    [[0.20, 0.85]],
    [[0.20, 0.95]],
    [[0.10, 0.95]]
], dtype=np.float32)

obstacles_list = [poly1_m_rand, poly3_m_rand]

# Example Search Parameters (Row, Column)
# (10, 10) is near the bottom-left corner
SearchStart_real = (20, 20) 
SearchStart = real_to_grid(SearchStart_real)
print("SearchStart (grid):", SearchStart)
# (180, 180) is near the top-right corner
SearchGoal_real = (80, 80)
SearchGoal = real_to_grid(SearchGoal_real)
print("SearchGoal (grid):", SearchGoal)

# 1. Create the Occupancy Map
# Grid cells: 0 = Free, -1 = Obstacle
Map = create_occupancy_grid(obstacles_list)
display_grid(Map)
# 2. Run the A* Search
path, explored = grid_search(Map, SearchStart, SearchGoal)

# 3. Process and Display Results
if path:
    simplified_path = simplify_path(path)
    
    print("--- A* Pathfinding Results ---")
    print(f"Robot Size: {robot_h}x{robot_w} cells")
    print(f"Start: {SearchStart}, Goal: {SearchGoal}")
    print(f"Total cells explored: {len(explored)}")
    print(f"Full Path Length (cells): {len(path)-1}")
    print(f"Simplified Path Waypoints: {len(simplified_path)}")
    print(simplified_path)
    converted_simplified_path = [grid_to_real(p) for p in simplified_path]
    print("Simplified Path Waypoints (real cm):", converted_simplified_path)
    print("------------------------------")

    display_map(Map, path, simplified_path, SearchStart, SearchGoal, explored)

else:
    print("No path found.")