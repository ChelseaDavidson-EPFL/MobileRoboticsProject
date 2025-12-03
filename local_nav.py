from thymio import Thymio
import utils
import numpy as np
import math
import utils

GLOBAL_IR_THLD = 4000
LOCAL_IR_THLD = 2000

K_AVOID  = 100
K_BREAK = 1200
FWD_SPEED = 150
ROT_SPEED = 90
MAX_IR_VAL = 5000
DIS_THLD = 2000
ADVENCE_DIST = utils.ROBOT_H

KIDNAP_THRESHOLD = 100  # Below this value = robot is lifted (no ground detected)
GROUND_THRESHOLD = 700  # Above this value = robot is back on ground

def is_object(thym: Thymio):
    ir_max = max(thym.ir_sensors)
    if(thym.nav_mode == "GLOBAL" and ir_max > GLOBAL_IR_THLD or
       thym.nav_mode == "LOCAL" and ir_max > LOCAL_IR_THLD):
        return True
    return False

def avoid_obstacle(thym: Thymio, avoid_right: bool,  stage=0, pos_at_obst=[], angle_at_obst=0):
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
        
#     thym.set_motor_speeds([speed_L, speed_R])
#     return


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

    if(max(thym.ground_sensors)<KIDNAP_THRESHOLD):
        # print("KIDNAPPED groud sensors:", thym.ground_sensors)
        return True
    else:
        # print("GROUND groud sensors:", thym.ground_sensors)
        return False

    # # Robot is lifted when ground sensors read LOW (no ground detected)
    # is_lifted = ground_sensors[0] < KIDNAP_THRESHOLD and ground_sensors[1] < KIDNAP_THRESHOLD

    # # Robot is on ground when sensors read HIGH
    # is_on_ground = ground_sensors[0] > GROUND_THRESHOLD and ground_sensors[1] > GROUND_THRESHOLD

    # if is_lifted and not thym.is_kidnapped:
    #     # Robot just got kidnapped
    #     thym.is_kidnapped = True
    #     thym.stop()
    #     print("KIDNAPPED: Robot lifted! Motors stopped.")
    #     return "kidnapped"

    # elif is_on_ground and thym.is_kidnapped:
    #     # Robot was put back on the ground
    #     thym.is_kidnapped = False
    #     print("RECOVERED: Robot back on ground. Ready to relaunch path finding.")
    #     return "recovered"

    return None
