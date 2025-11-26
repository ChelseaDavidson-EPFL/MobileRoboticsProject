from thymio import Thymio

def is_object():
    return False

def avoid_obstacle(thym: Thymio, grid, path):
    thym.set_motor_speeds([0,0])
    return 