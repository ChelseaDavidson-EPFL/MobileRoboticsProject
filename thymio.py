class Thymio :
    ir_sensors = [0, 0, 0, 0, 0, 0]
    motor_speeds = [0, 0]
    pos = [0, 0]
    orient = 0
    nav_mode = "GLOBAL"


    # Constructor
    def __init__(self, pos_init, orient):
        self.pos = pos_init
        self.orient = orient
        self.nav_mode = "GLOBAL"
        self.ir_sensors = self.get_ir()
        self.set_motor_speeds(0, 0)

    # Methods
    def set_motor_speeds(self, left_speed, right_speed):
        self.motor_speeds[0] = left_speed
        self.motor_speeds[1] = right_speed
        # TODO : Set the motor speeds on the Thymio robot


    def get_ir(self):
        # TODO : Get the IR sensor values from the Thymio robot
        return self.ir_sensors

    def set_pos(self, new_pos, orient):
        self.pos = new_pos
        self.orient = orient
        # TODO : Update the position on the Thymio robot if necessary
    
