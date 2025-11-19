import tdmclient.notebook
from tdmclient import ClientAsync, aw
import time


class Thymio :
    ir_sensors = [0, 0, 0, 0, 0]
    motor_speeds = [0, 0]
    pos = [0, 0]
    orient = 0
    nav_mode = "GLOBAL"


    # Constructor
    def __init__(self, pos_init, orient):
        self.client = ClientAsync()
        self.node = None
        self.pos = pos_init
        self.orient = orient
        self.nav_mode = "GLOBAL"
        self.ir_sensors = [0, 0, 0, 0, 0]
        self.motor_speeds = [0, 0]

    # Methods

    async def _connect_to_thymio_(self):
        self.node = await self.client.wait_for_node()
        print("Thymio connected")
        await self.node.lock()


    def set_motor_speeds(self, speeds):
        
        self.node.flush()
        self.motor_speeds = speeds
        self.node.send_set_variables({"motor.left.target": [speeds[0]], "motor.right.target": [speeds[1]]})


    async def update_ir(self):
        self.node.flush()
        await self.node.wait_for_variables({"prox.horizontal"})
        if "prox.horizontal" in self.node:
            self.ir_sensors = self.node["prox.horizontal"]
            self.ir_sensors = self.ir_sensors[0:5]

    def set_pos(self, new_pos, orient):
        self.pos = new_pos
        self.orient = orient
        # TODO : Update the position on the Thymio robot if necessary
    
    def stop(self):
        self.set_motor_speeds([0, 0])
    
