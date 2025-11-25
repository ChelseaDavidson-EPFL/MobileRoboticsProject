import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

#def heuristic(a, b):
    # Implement the Manhattan distance heuristic
    #return abs(a[0] - b[0]) + abs(a[1] - b[1])

def heuristic(a, b):
    #Octile distance (better for moving in an 8-cell-connected grid)
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)


def display_map(map_grid, path, start, goal, explored):
    cmap = ListedColormap(['white', 'black', 'blue', 'green', 'red', 'grey', 'yellow'])
    map_display = np.zeros_like(map_grid, dtype=object)

    # Assign colors based on the map grid values
    map_display[map_grid == -1] = 'black'  # Obstacles
    map_display[map_grid == 0] = 'white'   # Free space

    for position in explored:
        if map_display[tuple(position)] == 'white':
            map_display[tuple(position)] = 'grey'  # Explored cells

    # Visualize the path
    for position in path:
        if map_display[position[0], position[1]] in ['white', 'grey']:
            map_display[position[0], position[1]] = 'blue'  # Path

    map_display[map_grid > 0] = 'yellow'  # Obstacles
    map_display[start[0], start[1]] = 'green'  # Start
    map_display[goal[0], goal[1]] = 'red'      # Goal

    # Convert color names to numbers for plotting
    color_mapping = {'white': 0, 'black': 1, 'blue': 2, 'green': 3, 'red': 4, 'grey': 5, 'yellow': 6}
    map_numeric_display = np.vectorize(color_mapping.get)(map_display)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(map_numeric_display, cmap=cmap)
    ax.set_xticks(np.arange(-0.5, map_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, map_grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(f'{ALGO} Visualization')
    plt.show()

Map = np.zeros((11, 9))

def grid_search(map_grid, S, G):
    
    ## initialize the varibales above
    came_from = {}      # to reconstruct path
    g_costs = {S: 0}    # cost from start to the cell
    explored = set()    # to keep track of explored cells
    operation_count = 0 # to count the number of operations
    
    open_set = [(heuristic(S, G), 0, S)]  # priority queue for A* (f_cost, g_cost, position)
        
    while open_set: # G goal is unmarked
        current_f_cost, current_g_cost, current_pos = heappop(open_set)

        explored.add(current_pos)
        
        if current_pos == G: # if G goal is marked
            break

        # Get neighbors (up, down, left, right)
        neighbors = [
            (current_pos[0]-1, current_pos[1]),  # Up
            (current_pos[0]+1, current_pos[1]),  # Down
            (current_pos[0], current_pos[1]-1),  # Left
            (current_pos[0], current_pos[1]+1),   # Right

            # diagonals
            (current_pos[0]-1, current_pos[1]-1),
            (current_pos[0]-1, current_pos[1]+1),
            (current_pos[0]+1, current_pos[1]-1),
            (current_pos[0]+1, current_pos[1]+1),
            ]
        

        for neighbor in neighbors: # for each neighbor of marked cells
            # Check if neighbor is within bounds
            if (0 <= neighbor[0] < map_grid.shape[0]) and (0 <= neighbor[1] < map_grid.shape[1]): #check if the neighbor is less than 1 and greater than 0
                # Check if neighbor is not an obstacle
                if map_grid[neighbor[0], neighbor[1]] != -1:
                    # determine if diagonal
                    dx = abs(neighbor[0] - current_pos[0])
                    dy = abs(neighbor[1] - current_pos[1])

                    if (dx == 1 and dy == 1):
                        move_cost = np.sqrt(2) 
                    else:
                        move_cost = 1

                    tentative_g_cost = current_g_cost + move_cost + map_grid[neighbor[0], neighbor[1]]
                    tentative_g_cost = current_g_cost + 1  + map_grid[neighbor[0], neighbor[1]]

                    if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                        g_costs[neighbor] = tentative_g_cost
                        came_from[neighbor] = current_pos
                        operation_count += 1 
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
        return path, explored, operation_count  # Return reversed path and explored cells
    else:
    # If we reach here, no path was found
        return None, explored, operation_count


# A* results
path, explored, operation_count = grid_search(Map, SearchStart, SearchGoal)
if path:
    print("A* path length =", len(path)-1, "\n", path)
    print("length explored:", len(explored))
    print("A* visualization")
    display_map(Map, path, SearchStart, SearchGoal, explored)
else:
    print("No path found with A*")

