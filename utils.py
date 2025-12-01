# ============================================================
#  Constants
# ============================================================
GRID_DIM = 200
ROBOT_H = 11            # Robot height in cm (Thymio)
ROBOT_W = 11            # Robot width in cm (Thymio)
FREQ_MAIN_LOOP = 20     # Frequency of the main loop (in Hz)

 
# # ============================================================
#  GLOBAL VARIABLES (accessible by all files)
# ============================================================
          # Grid is always 200x200 cells
cell_size_cm = None     # Set by vision after arena detection

arena_width_cm = None   # Set by vision after arena detection
arena_height_cm = None  # Set by vision after arena detection

# ============================================================
#  AXIS CONVERSION FUNCTIONS
#  Grid: (row, col) where row 0 is TOP, row 199 is BOTTOM
#  Real: (x_cm, y_cm) where BL is origin (0,0)
# ============================================================

def cell_to_cm(cell_index):
    """Converts cell index (0-199) to distance in cm."""
    return cell_index * cell_size_cm

def cm_to_cell(cm_value):
    """Converts distance in cm to cell index."""
    return int(cm_value / cell_size_cm)

def real_to_grid(coord):
    """
    Converts real-world coordinates (x_cm, y_cm) to grid coordinates (row, col).
    BL origin in real -> row increases downward, col increases rightward in grid.
    """
    x_cm, y_cm = coord
    col = int(x_cm / cell_size_cm)
    row = GRID_DIM - 1 - int(y_cm / cell_size_cm)  # Flip y-axis
    # Clamp to valid range
    row = max(0, min(GRID_DIM - 1, row))
    col = max(0, min(GRID_DIM - 1, col))
    return (row, col)

def grid_to_real(coord):
    """
    Converts grid coordinates (row, col) to real-world coordinates (x_cm, y_cm).
    Returns center of the cell in cm.
    """
    row, col = coord
    x_cm = (col + 0.5) * cell_size_cm
    y_cm = (GRID_DIM - 1 - row + 0.5) * cell_size_cm
    return (x_cm, y_cm)
