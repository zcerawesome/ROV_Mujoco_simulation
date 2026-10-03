import numpy as np

# inputs array velocity [x, y, z, x_rot, y_rot, z_rot]

class Movement:

    def __init__(self):
        self.movement = [0, 0, 0, 0, 0, 0]
        self.current_movement = [0 for i in range(6)]

    def move(self, x, y, z, x_rot, y_rot, z_rot):
        self.movement = [x, y, z, x_rot, y_rot, z_rot]
        return [x, y, z, x_rot, y_rot, z_rot]

    def get_move(self):
        return self.movement