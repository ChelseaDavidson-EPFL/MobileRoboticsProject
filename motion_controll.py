import numpy as np
import math
from thymio import Thymio

# ============================================================
#  CONSTANTS FOR MOTION CONTROL
# ============================================================
GOAL_RADIUS = 2       # Radius around goal considered as reached (cm)
FWD_SPEED = 150         # Forward speed during path following
ROT_SPEED = 100         # Rotation speed when aligning
MAX_ANGLE = 0.8         # Angle threshold to start moving forward (rad)
K_ASTOL = 4000          # Astolfi controller gain for smooth turning
DIST_TO_WHEELS = 0.0475 # Distance from robot center to wheels (m)


# ============================================================
#  PATH FOLLOWING
# ============================================================
def follow_path(thym: Thymio, next_goal):
    """
    Controls robot movement toward the next waypoint using Astolfi controller.
    Combines rotation and forward motion for smooth trajectory following.

    Args:
        thym: Thymio robot instance with position and orientation
        next_goal: Target waypoint [x_cm, y_cm] in real coordinates

    Returns:
        bool: True if waypoint reached (within GOAL_RADIUS), False otherwise
    """
    
    # Calculate error from goal
    dx = next_goal[0] - thym.pos[0]
    dy = next_goal[1] - thym.pos[1]
    dist_goal = math.sqrt(dx**2 + dy**2)

    goal_reached = False
    # Check if goal is reached
    if(dist_goal < GOAL_RADIUS):
        goal_reached = True
    
    # Calculate angle to goal (0=X+, π/2=Y+, -π/2=Y-, π=X-)
    angle_goal = math.atan2(dy, dx)
    diff_angle = angle_goal - thym.orient
    
    # Normalize angle difference to [-π, π]
    if(diff_angle > math.pi):
        diff_angle -= 2*math.pi
    if(diff_angle < -math.pi):
        diff_angle += 2*math.pi

    # Rotate in place if angle error is too large
    if(abs(diff_angle) > MAX_ANGLE):
        if(diff_angle >= 0):
            thym.set_motor_speeds([-ROT_SPEED, ROT_SPEED])
        else:
            thym.set_motor_speeds([ROT_SPEED, -ROT_SPEED])
        return goal_reached
        
    # Astolfi controller: smooth turn while moving forward
    angle_speed = K_ASTOL * diff_angle * DIST_TO_WHEELS
    left_speed = int(FWD_SPEED - angle_speed)
    right_speed = int(FWD_SPEED + angle_speed)
    
    thym.set_motor_speeds([left_speed, right_speed])

    return goal_reached
        
