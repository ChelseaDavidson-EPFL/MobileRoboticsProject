from thymio import Thymio
import utils
import numpy as np
import math
import utils

# ============================================================
#  CONSTANTS FOR LOCAL NAVIGATION
# ============================================================
GLOBAL_IR_THLD = 4000   # IR threshold for obstacle detection in global navigation mode
LOCAL_IR_THLD = 2000    # IR threshold for obstacle detection in local navigation mode

K_AVOID  = 100          # Proportional gain for obstacle avoidance speed adjustment
K_BREAK = 1200          # Braking constant (unused)
FWD_SPEED = 150         # Forward speed during obstacle avoidance
ROT_SPEED = 90          # Rotation speed when turning
MAX_IR_VAL = 5000       # Maximum IR sensor value for normalization
DIS_THLD = 2000         # Distance threshold (unused)
ADVENCE_DIST = utils.ROBOT_H  # Distance to advance after clearing obstacle

KIDNAP_THRESHOLD = 100  # Below this value = robot is lifted (no ground detected)
GROUND_THRESHOLD = 700  # Above this value = robot is back on ground


# ============================================================
#  OBSTACLE DETECTION
# ============================================================
def is_object(thym: Thymio):
    """
    Checks if an obstacle is detected by the IR sensors.
    Uses different thresholds depending on navigation mode.

    Args:
        thym: Thymio robot instance with IR sensor data

    Returns:
        bool: True if obstacle detected, False otherwise
    """
    ir_max = max(thym.ir_sensors)
    if(thym.nav_mode == "GLOBAL" and ir_max > GLOBAL_IR_THLD or
       thym.nav_mode == "LOCAL" and ir_max > LOCAL_IR_THLD):
        return True
    return False


# ============================================================
#  OBSTACLE AVOIDANCE
# ============================================================
def avoid_obstacle(thym: Thymio, avoid_right: bool,  stage=0, pos_at_obst=[], angle_at_obst=0):
    """
    Executes a multi-stage obstacle avoidance maneuver.
    The robot either follows the left or right edge of an obstacle until it can resume global navigation.

    Stages:
        0: Rotate to align with obstacle edge
        1: Follow obstacle edge using wall-following behavior
        2: Advance forward after clearing the obstacle
        3: Rotate back towards original heading
        4: Advance a bit more and finish avoidance

    Args:
        thym: Thymio robot instance
        avoid_right: True to avoid right (keep obstacle on left), False to avoid left
        stage: Current stage of the avoidance maneuver (0-4)
        pos_at_obst: Position where obstacle was detected [x_cm, y_cm]
        angle_at_obst: Orientation when obstacle was detected

    Returns:
        tuple: (obstacle_avoided, stage, pos_at_obst, angle_at_obst)
            - obstacle_avoided: True if avoidance maneuver is complete
            - stage: Updated stage number
            - pos_at_obst: Updated position at obstacle
            - angle_at_obst: Updated angle at obstacle
    """
    ir_sens = thym.ir_sensors
    last_angle = thym.last_orient
    print(f"stage: {stage}, IR: {ir_sens}")
    
    if(avoid_right):
        # Avoid RIGHT: keep obstacle on LEFT (sensor 0)
        match stage:
            case 0:
                if(ir_sens[0]==0 or sum(ir_sens[1:5])>0):
                    thym.set_motor_speeds([ROT_SPEED, -ROT_SPEED])
                    print(f"IR left: {ir_sens[0]}")
                else:
                    stage=1
            case 1:
                if(sum(ir_sens[0:5])<=0):
                    pos_at_obst = thym.pos.copy()
                    stage=2
                else:
                    if(ir_sens[0] > max(ir_sens[1:5])):
                        thym.set_motor_speeds([FWD_SPEED, int(FWD_SPEED-K_AVOID*max(ir_sens[1:5])/MAX_IR_VAL)])
                    else:
                        thym.set_motor_speeds([int(FWD_SPEED-K_AVOID*ir_sens[0]/MAX_IR_VAL), FWD_SPEED])
            case 2:
                thym.set_motor_speeds([FWD_SPEED, FWD_SPEED])
                dis_from_obst = math.sqrt((thym.pos[0]-pos_at_obst[0])**2 + (thym.pos[1]-pos_at_obst[1])**2)
                if(dis_from_obst>=4*ADVENCE_DIST/3):
                    stage=3
                    angle_at_obst = thym.orient
            case 3:
                thym.set_motor_speeds([-ROT_SPEED, ROT_SPEED])
                if(ir_sens[0]>0):
                    pos_at_obst = thym.pos.copy()
                    stage=4
            case 4: 
                thym.set_motor_speeds([FWD_SPEED, FWD_SPEED])
                dis_from_obst = math.sqrt((thym.pos[0]-pos_at_obst[0])**2 + (thym.pos[1]-pos_at_obst[1])**2)
                if(dis_from_obst>=2*ADVENCE_DIST/3):
                    thym.stop()
                    return True, 0, [0,0], 0
    else:
        # Avoid LEFT: keep obstacle on RIGHT (sensor 4)
        match stage:
            case 0:
                if(ir_sens[4]==0 or sum(ir_sens[0:4])>0):
                    thym.set_motor_speeds([-ROT_SPEED, ROT_SPEED])
                    print(f"IR right: {ir_sens[4]}")
                else:
                    stage=1
            case 1:
                if(sum(ir_sens[0:5])<=0):
                    pos_at_obst = thym.pos.copy()
                    stage=2
                else:
                    if(ir_sens[4] > max(ir_sens[0:4])):
                        thym.set_motor_speeds([int(FWD_SPEED-K_AVOID*max(ir_sens[0:4])/MAX_IR_VAL), FWD_SPEED])
                    else:
                        thym.set_motor_speeds([FWD_SPEED, int(FWD_SPEED-K_AVOID*ir_sens[4]/MAX_IR_VAL)])
            case 2:
                thym.set_motor_speeds([FWD_SPEED, FWD_SPEED])
                dis_from_obst = math.sqrt((thym.pos[0]-pos_at_obst[0])**2 + (thym.pos[1]-pos_at_obst[1])**2)
                if(dis_from_obst>=4*ADVENCE_DIST/3):
                    stage=3
                    angle_at_obst = thym.orient
            case 3:
                thym.set_motor_speeds([ROT_SPEED, -ROT_SPEED])
                if(ir_sens[4]>0):
                    pos_at_obst = thym.pos.copy()
                    stage=4
            case 4: 
                thym.set_motor_speeds([FWD_SPEED, FWD_SPEED])
                dis_from_obst = math.sqrt((thym.pos[0]-pos_at_obst[0])**2 + (thym.pos[1]-pos_at_obst[1])**2)
                if(dis_from_obst>=2*ADVENCE_DIST/3):
                    thym.stop()
                    return True, 0, [0,0], 0
    
    return False, stage, pos_at_obst, angle_at_obst


# ============================================================
#  OBSTACLE SIDE DETECTION
# ============================================================
def avoid_right(thym: Thymio, grid):
    """
    Analyzes the area in front of the robot to determine which side has more obstacles.
    Used to decide whether to follow the left or right edge of an obstacle.

    The function scans a rectangular region ahead of the robot, counting obstacles
    on each side. The robot should avoid towards the side with fewer obstacles.

    Args:
        thym: Thymio robot instance with position and orientation
        grid: Occupancy grid (0=free, -1=obstacle)

    Returns:
        bool: True if robot should avoid right (more obstacles on left),
              False if robot should avoid left (more obstacles on right)
    """
    # Convert Thymio position to grid coordinates
    pos_cm = (thym.pos[0], thym.pos[1])
    grid_pos = utils.real_to_grid(pos_cm)
    row, col = grid_pos
    
    # Determine right/left direction based on robot orientation
    # orient: 0=right (X+), π/2=up (Y+), -π/2=down (Y-), π=left (X-)
    orient = thym.orient
    
    # Forward direction in grid coordinates
    # In real coords: forward = (cos(θ), sin(θ))
    # In grid coords: row increases downward, col increases rightward
    # So we need to invert the row component
    fwd_col = int(round(math.cos(orient)))  # X component
    fwd_row = -int(round(math.sin(orient)))  # -Y component (grid rows are inverted)
    
    # Right vector perpendicular to forward (rotation of -90° in real coords)
    # right = (sin(θ), -cos(θ)) in real coords
    # In grid: right_col = sin(θ), right_row = cos(θ)
    right_col = int(round(math.sin(orient)))
    right_row = int(round(math.cos(orient)))
    
    # Left vector (opposite of right)
    left_col = -right_col
    left_row = -right_row
    
    # Check an area in front of the robot (both sides and forward)
    lateral_distance = 50  # How far to check sideways
    forward_depth = 30     # How far to check forward
    obstacles_right = 0
    obstacles_left = 0
    
    # print(f"Robot at grid ({row}, {col}), orient={orient:.2f}, fwd=({fwd_row},{fwd_col}), right=({right_row},{right_col})")
    
    # Scan an area: for each forward distance, check cells to the left and right
    for fwd_dist in range(0, forward_depth + 1):
        for lateral_dist in range(1, lateral_distance + 1):
            # Base position at this forward distance
            base_row = row + fwd_row * fwd_dist
            base_col = col + fwd_col * fwd_dist
            
            # Cell to the right of this forward position
            right_row_check = base_row + right_row * lateral_dist
            right_col_check = base_col + right_col * lateral_dist
            if 0 <= right_row_check < grid.shape[0] and 0 <= right_col_check < grid.shape[1]:
                if grid[right_row_check, right_col_check] == -1:
                    obstacles_right += 1
            
            # Cell to the left of this forward position
            left_row_check = base_row + left_row * lateral_dist
            left_col_check = base_col + left_col * lateral_dist
            if 0 <= left_row_check < grid.shape[0] and 0 <= left_col_check < grid.shape[1]:
                if grid[left_row_check, left_col_check] == -1:
                    obstacles_left += 1
    
    # Return True if more obstacles on the left (so avoid to the right)
    print(f"Obstacles in front - Left: {obstacles_left}, Right: {obstacles_right}")
    return obstacles_left >= obstacles_right


# ============================================================
#  KIDNAP DETECTION
# ============================================================
def check_kidnap(thym: Thymio):
    """
    Detects if the robot has been lifted off the ground (kidnapped).
    Uses ground sensors to detect absence of surface underneath.

    Args:
        thym: Thymio robot instance with ground sensor data

    Returns:
        bool: True if robot is lifted (kidnapped), False otherwise
    """
    ground_sensors = thym.get_ground_sensors()

    # Check if the robot is lifted (kidnapped)
    if(max(thym.ground_sensors)<KIDNAP_THRESHOLD):
        return True
    else:
        return False
    return None
