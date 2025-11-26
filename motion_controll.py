import numpy as np
import math
from thymio import Thymio

#radius around witch the goal is reached
GOAL_RADUIS = 2
# Forward speed
FWD_SPEED = 500
# angle at witch it starts to go forward
MAX_ANGLE = 0.5
# Astolfi angle gain
K_ASTOL = 9000
# Distance between the center of the tymio and the wheels
DIST_TO_WHEELS = 0.0475

def follow_path(thym: Thymio, next_goal):
    
    # Calculate error form goal
    dx = next_goal[0] - thym.pos[0]
    dy = next_goal[1] - thym.pos[1]
    dist_goal = math.sqrt(dx**2+dy**2)

    # retrun true if goal is reached
    if(dist_goal<GOAL_RADUIS):
        return True
    
    
    angle_goal = math.atan2(dx, dy)
    diff_angle = angle_goal - thym.orient

    if(abs(diff_angle)>MAX_ANGLE):
        if(diff_angle>=0):
            thym.set_motor_speeds([100, -100])
        else:
            thym.set_motor_speeds([-100, 100])
        
    # Astolfi implementation
    angle_speed = K_ASTOL * diff_angle * DIST_TO_WHEELS
    left_speed = int(FWD_SPEED - angle_speed)
    right_speed = int(FWD_SPEED + angle_speed)
    
    print(f"angle_speed: {angle_speed}, left_speed: {left_speed}, right_speed: {right_speed}")

    thym.set_motor_speeds([left_speed, right_speed])

    return False
        
