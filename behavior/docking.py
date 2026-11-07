# Nick's File
import math
import cv_processor as cvp

class Docking:
    def __init__(self, maximum) -> None:
        self.stage = 0
        self.maxiumum = maximum
        self.threshold = 0.1

    def meet_threshold(self, x, y):
        return math.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) <= self.threshold


    def forward_process(self, frame, x, y, w, h):
        if w * h >= self.maxiumum:
            self.stage = 1 # Nick's contribution (10/24/26)
            return (0, 0, 0, 0, 0, 0)

        if self.meet_threshold(x, y):
            return (0.5, 0, 0, 0, 0, 0)
        else:
            # front camera: image right = body -y, image down = body -z
            return (0, -(x - 0.5) * 2, -(y - 0.5) * 2, 0, 0, 0)


    def down_process(self, frame, x, y, w, h):
        if w * h >= self.maxiumum:
            return (0, 0, 0, 0, 0, 0)
        if self.meet_threshold(x, y):
            return (0, 0, -0.5, 0, 0, 0)
        else:
            return (-(y - 0.5) * 2, -(x - 0.5) * 2, 0, 0, 0, 0)

    def process(self, frames):
        front_view, x1, y1, w1, h1 = cvp.process_pipe(frames['front'])
        down_view, x2, y2, w2, h2 = cvp.process_pipe(frames['down'])
        f_area = w1 * h1
        d_area = w2 * h2

        if self.stage == 0:
            return self.forward_process(front_view, x1, y1, w1, h1)
        elif self.stage == 1:
            if f_area > 0:
                return (0, 0, 1, 0, 0, 0)
            else:
                self.stage = 2
        elif self.stage == 2:
            if d_area < self.threshold:
                return (0.5, 0, 0, 0, 0, 0)
            else:
                self.stage = 3
        elif self.stage == 3:
            return self.down_process(down_view, x2, y2, w2, h2)
        
        return (0, 0, 0, 0, 0, 0)