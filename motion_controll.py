import numpy as np
import math
from thymio import Thymio

#radius around witch the goal is reached (in cm)
GOAL_RADUIS = 2.5
# Forward speed
FWD_SPEED = 150
# Rotation speed
ROT_SPEED = 100
# angle at witch it starts to go forward
MAX_ANGLE = 0.8
# Astolfi angle gain
K_ASTOL = 4000
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
    
    
    # angle_goal: 0=X+, π/2=Y+, -π/2=Y-, π=X-
    angle_goal = math.atan2(dy, dx)
    diff_angle = angle_goal - thym.orient
    if(diff_angle>math.pi):
        diff_angle -= 2*math.pi
    if(diff_angle<-math.pi):
        diff_angle += 2*math.pi

    # print(f"pos: [{thym.pos[0]:.1f}, {thym.pos[1]:.1f}], goal: [{next_goal[0]:.1f}, {next_goal[1]:.1f}], dist: {dist_goal:.2f}, angle_goal: {angle_goal:.2f}, orient: {thym.orient:.2f}, diff: {diff_angle:.2f}")

    if(abs(diff_angle)>MAX_ANGLE):
        if(diff_angle>=0):
            thym.set_motor_speeds([-ROT_SPEED, ROT_SPEED])
        else:
            thym.set_motor_speeds([ROT_SPEED, -ROT_SPEED])
        return False
        
    # Astolfi implementation
    angle_speed = K_ASTOL * diff_angle * DIST_TO_WHEELS
    left_speed = int(FWD_SPEED - angle_speed)
    right_speed = int(FWD_SPEED + angle_speed)
    
    # print(f"angle_speed: {angle_speed}, left_speed: {left_speed}, right_speed: {right_speed}")

    thym.set_motor_speeds([left_speed, right_speed])

    return False
        
