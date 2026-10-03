class Spin:
    def __init__(self, target_deg=360.0, rate=0.5):
        self.target, self.rate = target_deg, rate
        self.prev_yaw = None
        self.total = 0.0
        self.task_complete = False

    def current_pos(self, x, y, z, roll, pitch, yaw):
        if self.prev_yaw is not None:
            d = (roll - self.prev_yaw + 180.0) % 360.0 - 180.0   # shortest signed step
            self.total += d
        self.prev_yaw =roll 

    def complete_task(self, state=True):
        self.task_complete = state

    def done(self):
        return abs(self.total) >= self.target or self.task_complete

    def current_move(self):
        return (0, 0, 0, 0, 0, 0) if self.done() else (0, 0, 0, 0, 0, self.rate)
