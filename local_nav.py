from thymio import Thymio
import utils
import numpy as np
import math
import utils

GLOBAL_IR_THLD = 5000
LOCAL_IR_THLD = 2000

K_AVOID  = 1
K_BREAK = 1200
FWD_SPEED = 50
ROT_SPEED = 30
MAX_IR_VAL = 5000
DIS_THLD = 2000

KIDNAP_THRESHOLD = 300  # Below this value = robot is lifted (no ground detected)
GROUND_THRESHOLD = 700  # Above this value = robot is back on ground

def is_object(thym: Thymio):
    ir_max = max(thym.ir_sensors)
    if(thym.nav_mode == "GLOBAL" and ir_max > GLOBAL_IR_THLD or
       thym.nav_mode == "LOCAL" and ir_max > LOCAL_IR_THLD):
        return True
    return False


def avoid_right(thym: Thymio, grid):
    """
    Checks if there are more obstacles to the right or left of the robot.
    Returns True if the robot should avoid to the right (obstacle on the left).
    """
    # Convert Thymio position to grid coordinates
    pos_cm = (thym.pos[0], thym.pos[1])
    grid_pos = utils.real_to_grid(pos_cm)
    row, col = grid_pos
    
    # Determine right/left direction based on robot orientation
    # orient: 0=right (X+), π/2=up (Y+), -π/2=down (Y-), π=left (X-)
    orient = thym.orient
    
    # Right vector perpendicular to orientation (rotation of -90°)
    right_dx = int(round(math.sin(orient)))
    right_dy = int(round(-math.cos(orient)))
    
    # Left vector (rotation of +90°)
    left_dx = -right_dx
    left_dy = -right_dy
    
    # Check cells to the right and left (check distance: 3 cells)
    check_distance = 3
    obstacles_right = 0
    obstacles_left = 0
    
    for dist in range(1, check_distance + 1):
        # Cell to the right
        right_row = row + right_dx * dist
        right_col = col + right_dy * dist
        if 0 <= right_row < grid.shape[0] and 0 <= right_col < grid.shape[1]:
            if grid[right_row, right_col] == -1:
                obstacles_right += 1
        
        # Cell to the left
        left_row = row + left_dx * dist
        left_col = col + left_dy * dist
        if 0 <= left_row < grid.shape[0] and 0 <= left_col < grid.shape[1]:
            if grid[left_row, left_col] == -1:
                obstacles_left += 1
    
    # Return True if more obstacles on the left (so avoid to the right)
    return obstacles_left >= obstacles_right
    
    

def avoid_obstacle(thym: Thymio, avoid_right: bool, start_angle):
    ir_sens = thym.ir_sensors
    last_angle = thym.last_orient
    if(avoid_right):
        if(ir_sens[0]==0 or sum(ir_sens[1:5])>0):
            thym.set_motor_speeds([ROT_SPEED, -ROT_SPEED])
        else :
            print(f"IR left: {ir_sens[0]}")
            thym.set_motor_speeds([int(FWD_SPEED - K_AVOID*(ir_sens[0]-DIS_THLD)), FWD_SPEED])
    else:
        if(ir_sens[4]==0 or sum(ir_sens[0:4])>0):
            thym.set_motor_speeds([-ROT_SPEED, ROT_SPEED])
        else :
            thym.set_motor_speeds([FWD_SPEED, int(FWD_SPEED - K_AVOID*(ir_sens[4]-DIS_THLD))])

    if(abs(thym.orient - start_angle) > math.pi/4):
        return True
    return False

# def avoid_obstacle(thym: Thymio, grid, avoid_right: bool):
#     ir_sens = thym.ir_sensors
#     left_sum = ir_sens[0]+ir_sens[1]
#     right_sum = ir_sens[3]+ir_sens[4]
#     fwd_speed = int(FWD_SPEED - K_BREAK*((ir_sens[1]+ir_sens[2]+ir_sens[3])/3*MAX_IR_VAL))

#     if(fwd_speed<=0): 
#         fwd_speed = 0
            
#     speed_L = fwd_speed
#     speed_R = fwd_speed

#     if(fwd_speed == 0 and right_sum == 0 and left_sum == 0):
#         if(avoid_right):
#             left_sum = 1000
#         else:
#             right_sum = 1000
        

#     if(left_sum > right_sum):
#         speed_R = int(speed_R - (left_sum/MAX_IR_VAL)*K_AVOID)      
#     else: 
#         speed_L = int(speed_L - (right_sum/MAX_IR_VAL)*K_AVOID) 
        
    thym.set_motor_speeds([speed_L, speed_R])
    return


def check_kidnap(thym: Thymio):
    """
    Check if the robot has been kidnapped (lifted off the ground).
    Ground sensors return HIGH values when close to ground, LOW values when lifted.

    Returns:
        - "kidnapped" if robot was just lifted (transition to kidnapped state)
        - "recovered" if robot was put back on ground (transition from kidnapped)
        - None if no state change
    """
    ground_sensors = thym.get_ground_sensors()

    # Robot is lifted when ground sensors read LOW (no ground detected)
    is_lifted = ground_sensors[0] < KIDNAP_THRESHOLD and ground_sensors[1] < KIDNAP_THRESHOLD

    # Robot is on ground when sensors read HIGH
    is_on_ground = ground_sensors[0] > GROUND_THRESHOLD and ground_sensors[1] > GROUND_THRESHOLD

    if is_lifted and not thym.is_kidnapped:
        # Robot just got kidnapped
        thym.is_kidnapped = True
        thym.stop()
        print("KIDNAPPED: Robot lifted! Motors stopped.")
        return "kidnapped"

    elif is_on_ground and thym.is_kidnapped:
        # Robot was put back on the ground
        thym.is_kidnapped = False
        print("RECOVERED: Robot back on ground. Ready to relaunch path finding.")
        return "recovered"

    return None
