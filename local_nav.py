from thymio import Thymio

GLOBAL_IR_THLD = 5000
LOCAL_IR_THLD = 3000

K_AVOID  = 800
K_BREAK = 1800
FWD_SPEED = 200
MAX_IR_VAL = 5000


def is_object(thym: Thymio):
    ir_sum = sum(thym.ir_sensors)
    if(thym.nav_mode == "GLOBAL" and ir_sum > GLOBAL_IR_THLD or
       thym.nav_mode == "LOCAL" and ir_sum > LOCAL_IR_THLD):
        return True
    return False


def avoid_right(thym: Thymio, grid):
    
    return True
    
    

def avoid_obstacle(thym: Thymio, grid, path, avoid_right: bool):
    ir_sens = thym.ir_sensors
    left_sum = ir_sens[0]+ir_sens[1]
    right_sum = ir_sens[3]+ir_sens[4]
    fwd_speed = int(FWD_SPEED - K_BREAK*((ir_sens[1]+ir_sens[2]+ir_sens[3])/3*MAX_IR_VAL))

    if(fwd_speed<=0): 
        fwd_speed = 0
            
    speed_L = fwd_speed
    speed_R = fwd_speed

    if(fwd_speed == 0 and right_sum == 0 and left_sum == 0):
        if(avoid_right):
            left_sum = 2000
        else:
            right_sum = 2000
        

    if(left_sum > right_sum):
        speed_R = int(speed_R - (left_sum/MAX_IR_VAL)*K_AVOID)      
    else: 
        speed_L = int(speed_L - (right_sum/MAX_IR_VAL)*K_AVOID) 
        
    thym.set_motor_speeds([speed_L, speed_R])
    return