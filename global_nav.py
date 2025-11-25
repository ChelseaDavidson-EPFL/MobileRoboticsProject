import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from heapq import heappush, heappop

# ============================================================
#  CONSTANTS
# ============================================================
# Robot footprint (width × height), centered on the path cell
robot_w = 3
robot_h = 3

valid=True

# ============================================================
#  HEURISTIC (Octile Distance)
# ============================================================
def heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)


# ============================================================
#  ROBOT COLLISION CHECK (center-based)
# ============================================================
def is_robot_valid(map_grid, cx, cy, robot_h, robot_w):
    """
    Checks whether a robot of size robot_h × robot_w centered at (cx, cy)
    fits entirely inside free space.
    Works for any even or odd robot size.
    """

    half_h = robot_h // 2
    half_w = robot_w // 2

    row_start = cx - half_h
    row_end   = cx + (robot_h - half_h - 1)
    
    col_start = cy - half_w
    col_end   = cy + (robot_w - half_w - 1)

    # Boundary check
    if row_start < 0 or col_start < 0 or row_end >= map_grid.shape[0] or col_end >= map_grid.shape[1]:
        return False

    # Obstacle check
    for r in range(row_start, row_end + 1):
        for c in range(col_start, col_end + 1):
            if map_grid[r, c] == -1:
                return False

    return True


# ============================================================
#  VISUALISATION 
# ============================================================
def display_map(map_grid, path, start, goal, explored):
    cmap = ListedColormap(['white', 'black', 'blue', 'green', 'red', 'grey', 'yellow'])
    map_display = np.zeros_like(map_grid, dtype=object)

    # Colors
    map_display[map_grid == -1] = 'black'
    map_display[map_grid == 0] = 'white'
    map_display[map_grid > 0] = 'yellow'

    # Explored cells
    for position in explored:
        if map_display[position] == 'white':
            map_display[position] = 'grey'

    # Path
    for position in path:
        if map_display[position] in ['white', 'grey']:
            map_display[position] = 'blue'

    # Start and goal
    map_display[start] = 'red'
    map_display[goal] = 'green'

    # Convert color names to numbers
    color_mapping = {'white': 0, 'black': 1, 'blue': 2,
                     'green': 3, 'red': 4, 'grey': 5, 'yellow': 6}
    map_numeric_display = np.vectorize(color_mapping.get)(map_display)

    # Show map
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(map_numeric_display, cmap=cmap)
    ax.set_xticks(np.arange(-0.5, map_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, map_grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.title(f"A* Pathfinding ({robot_h}×{robot_w} Robot)")

    # Draw robot footprint at each path cell
    half_h = robot_h / 2
    half_w = robot_w / 2

    for (cx, cy) in path:
        row_start = cx - half_h
        col_start = cy - half_w

        rect = plt.Rectangle(
            (col_start, row_start),
            robot_w,
            robot_h,
            linewidth=1.2,
            edgecolor='cyan',
            facecolor='cyan',
            alpha=0.25
        )
        ax.add_patch(rect)

    plt.show()


# ============================================================
#  A* SEARCH
# ============================================================
def grid_search(map_grid, S, G):

    came_from = {}
    g_costs = {S: 0}
    explored = set()

    open_set = [(heuristic(S, G), 0, S)]  # (f_cost, g_cost, position)

    while open_set:
        current_f_cost, current_g_cost, current_pos = heappop(open_set)
        explored.add(current_pos)

        if current_pos == G:
            break

        # 8-connected neighbors
        neighbors = [
            (current_pos[0]-1, current_pos[1]),     # Up
            (current_pos[0]+1, current_pos[1]),     # Down
            (current_pos[0], current_pos[1]-1),     # Left
            (current_pos[0], current_pos[1]+1),     # Right
            (current_pos[0]-1, current_pos[1]-1),   # diag
            (current_pos[0]-1, current_pos[1]+1),
            (current_pos[0]+1, current_pos[1]-1),
            (current_pos[0]+1, current_pos[1]+1)
        ]

        for neighbor in neighbors:
            nx, ny = neighbor

            # Bounds
            if not (0 <= nx < map_grid.shape[0] and 0 <= ny < map_grid.shape[1]):
                continue

            # --- robot collision check ---
            if not is_robot_valid(map_grid, nx, ny, robot_h, robot_w):
                continue

            # Move cost
            dx = abs(nx - current_pos[0])
            dy = abs(ny - current_pos[1])
            move_cost = np.sqrt(2) if (dx == 1 and dy == 1) else 1

            tentative_g_cost = current_g_cost + move_cost + map_grid[nx, ny]

            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                came_from[neighbor] = current_pos
                f_cost = tentative_g_cost + heuristic(neighbor, G)
                heappush(open_set, (f_cost, tentative_g_cost, neighbor))

    # reconstruct path
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
#  CREATE 50×50 MAP
# ============================================================
Map = np.zeros((50, 50))

# Borders
Map[0, :] = -1
Map[-1, :] = -1
Map[:, 0] = -1
Map[:, -1] = -1

# Obstacles
Map[10:40, 5] = -1
Map[20, 10:30] = -1
Map[30:40, 30:45] = -1
Map[5:15, 25] = -1
Map[15, 20:26] = -1

# Weighted cell
Map[25, 25] = 2

# Start & Goal
SearchStart = (40, 10)
SearchGoal = (5, 40)

# Validate robot positions
if not is_robot_valid(Map, SearchStart[0], SearchStart[1], robot_h, robot_w):
    print(f"ERROR: Start is invalid for {robot_h}×{robot_w} robot")
    valid=False

if not is_robot_valid(Map, SearchGoal[0], SearchGoal[1], robot_h, robot_w):
    print(f"ERROR: Goal is invalid for {robot_h}×{robot_w} robot")
    valid=False

# ============================================================
#  RUN A*
# ============================================================
path, explored = grid_search(Map, SearchStart, SearchGoal)

if path and valid:
    print("A* path length =", len(path)-1)
    print(path)
    display_map(Map, path, SearchStart, SearchGoal, explored)
else:
    print("No path found.")
