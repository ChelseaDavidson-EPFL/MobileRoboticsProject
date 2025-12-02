import asyncio
import tdmclient.notebook
from tdmclient import ClientAsync, aw
import time


class Thymio :
    ir_sensors = [0, 0, 0, 0, 0]
    motor_speeds = [0, 0]
    pos = [0, 0]
    orient = 0
    nav_mode = "GLOBAL" # navigation mode: "GLOBAL" or "LOCAL"
    FORWARD = 1
    DELTA_T = 4  # seconds forward
    SAMPLING = 0.1  # timer loop period

    # Button states
    button_forward = 0
    button_center = 0
    button_backward = 0

    # Constructor
    def __init__(self, pos_init, orient):
        self.client = ClientAsync()
        self.node = None
        self.pos = pos_init
        self.orient = orient
        self.last_orient = orient
        self.nav_mode = "GLOBAL"
        self.ir_sensors = [0, 0, 0, 0, 0]
        self.motor_speeds = [0, 0]
        self.state = 0
        self._forward_start_time = None
        self._timer_task = None

        # Button states
        self.button_forward = 0
        self.button_center = 0
        self.button_backward = 0

    # Methods

    async def _connect_to_thymio_(self):
        self.node = await self.client.wait_for_node()
        print("Thymio connected")
        await self.node.lock()

    async def unlock(self):
        if self.node is not None:
            await self.node.unlock()
            print("Thymio unlocked")

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


    async def update_buttons(self): # Read the current values of several buttons

        self.node.flush()
        await self.node.wait_for_variables({"button.forward", "button.center", "button.backward"})

        # Forward button
        if "button.forward" in self.node:
            val = self.node["button.forward"]
            if isinstance(val, list):
                val = val[0]
            self.button_forward = val
        else:
            self.button_forward = 0

        # Center button
        if "button.center" in self.node:
            val = self.node["button.center"]
            if isinstance(val, list):
                val = val[0]
            self.button_center = val
        else:
            self.button_center = 0
        
        # Backward button
        if "button.backward" in self.node:
            val = self.node["button.backward"]
            if isinstance(val, list):
                val = val[0]
            self.button_backward = val
        else:
            self.button_backward = 0

    async def button_loop(self):
        #"""Infinite loop reacting to the button values"""
        while True:
            await self.update_buttons()

            # Forward pressed → start moving
            if self.button_forward and self.state != self.FORWARD:
                print("Forward pressed")
                self.state = self.FORWARD
                self._forward_start_time = time.time()
                self.set_motor_speeds([100,100])

            # Center pressed → stop immediately
            if self.button_center:
                if self.state != 0:
                    print("Center pressed")
                self.state = 0
                self._forward_start_time = None
                self.set_motor_speeds([0,0])

            # Backward pressed → stop the loop and so this function
            if self.button_backward:
                print("Backward pressed")
                break

            # Automatically stop after DELTA_T seconds
            if self.state == self.FORWARD:
                elapsed = time.time() - self._forward_start_time
                if elapsed >= self.DELTA_T:
                    self.state = 0
                    self._forward_start_time = None
                    self.set_motor_speeds([0,0])

            await asyncio.sleep(self.SAMPLING)