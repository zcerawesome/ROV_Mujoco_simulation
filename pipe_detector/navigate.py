"""Steer toward the pipe from the detector's bounding box, and a closed-loop
check that this actually gets the ROV there.

ApproachPipe follows behavior/spin.py's shape: feed it detections, read back
a body-frame (x, y, z, roll, pitch, yaw) velocity command.

    python -m pipe_detector.navigate --episodes 30      # closed-loop success rate
"""
import argparse

import numpy as np
import mujoco as mj

from pipe_detector import underwater as uw
from pipe_detector.generate import FLOOR_Z, quat


class ApproachPipe:
    """Search -> turn to face the pipe -> close in on it with the front camera
    -> once it passes under the nose, centre it with the down camera and stop.
    Gains are per unit of normalised box offset; current_move is called once
    per control step."""

    def __init__(self, speed=0.4, yaw_gain=1.2, heave_gain=0.4, search_rate=0.5,
                 slow_size=0.12, down_gain=0.5, arrive=0.25, coast_steps=40, settle_steps=5):
        self.speed, self.yaw_gain, self.heave_gain = speed, yaw_gain, heave_gain
        self.search_rate, self.slow_size = search_rate, slow_size
        self.down_gain, self.arrive, self.coast_steps = down_gain, arrive, coast_steps
        self.settle_steps = settle_steps
        self.coast = 0              # steps left to keep going after the pipe left the front view
        self.settled = 0            # consecutive steps with the pipe centred below
        self.seen = False
        self.task_complete = False

    def done(self):
        return self.task_complete

    def current_move(self, front, down=None):
        """front/down: PipeDetector.detect() results (or None). Returns a
        body-frame (x, y, z, roll, pitch, yaw) velocity command."""
        if self.task_complete:
            return (0, 0, 0, 0, 0, 0)
        if down is not None:
            # pipe below: centre it. Down image up = ROV forward, image right = ROV starboard (-y)
            self.seen = True
            cx, cy = down["center"]
            # one centred frame could be a false detection: it has to stay centred
            self.settled = self.settled + 1 if max(abs(cx), abs(cy)) < self.arrive else 0
            if self.settled >= self.settle_steps:
                self.task_complete = True
                return (0, 0, 0, 0, 0, 0)
            return (-self.down_gain * cy, -self.down_gain * cx, 0, 0, 0, 0)
        self.settled = 0
        if front is None:
            if self.coast > 0:      # it slid out of the bottom of the view: it is just ahead, below
                self.coast -= 1
                return (0.5 * self.speed, 0, 0, 0, 0, 0)
            return (0, 0, 0, 0, 0, self.search_rate)              # spin until it is in view
        self.seen = True
        cx, cy = front["center"]
        # remember whether it was low in the view: if it vanishes next, keep going
        self.coast = self.coast_steps if cy > 0.3 else 0
        yaw = -self.yaw_gain * cx                                  # pipe right of centre -> turn right
        heave = -self.heave_gain * cy                              # pipe below centre -> go down
        # only drive forward once roughly facing it, and ease off as it fills the view
        facing = max(0.0, 1 - abs(cx) / 0.5)
        forward = self.speed * facing * max(0.3, 1 - front["size"] / self.slow_size)
        return (forward, 0, heave, 0, 0, yaw)


# ---------- closed-loop check ----------
def run_episode(gen, det, rng, water=True, steps=500, dt=0.1, min_height=0.35):
    """Drive the ROV with ApproachPipe on detector output alone, moving it
    kinematically (dt seconds per step). Returns (arrived, metres from the
    pipe at the end, steps used, whether it ever saw the pipe)."""
    gen._randomize_scene(rng)
    mj.mj_forward(gen.m, gen.d)
    pipe = gen._pipe_points(rng, 400)
    # start 2.5-5 m away from the pipe, facing a random direction
    start = pipe[rng.integers(len(pipe))][:2] + rng.uniform(2.5, 5.0) * np.array(
        [np.cos(a := rng.uniform(0, 2 * np.pi)), np.sin(a)])
    pos = np.array([*start, FLOOR_Z + rng.uniform(1.0, 2.2)])
    yaw = rng.uniform(0, 2 * np.pi)
    water_seed = int(rng.integers(1 << 31))
    nav = ApproachPipe()

    def look(cam, step):
        rgb, dist, cov = gen.render(cam)
        if water:   # same water all episode; fish, snow and sensor noise change every frame
            rgb, _, _ = uw.underwater(rgb, dist, cov, np.random.default_rng(water_seed),
                                      np.random.default_rng((water_seed, step, cam == "down")))
        return det.detect(rgb)

    for step in range(steps):
        gen.d.qpos[:3], gen.d.qpos[3:7] = pos, quat(yaw, 0, 0)
        mj.mj_forward(gen.m, gen.d)
        fx, fy, fz, _, _, wz = nav.current_move(look("front", step), look("down", step))
        if nav.done():
            break
        yaw += wz * dt
        c, s = np.cos(yaw), np.sin(yaw)
        pos += dt * np.array([fx * c - fy * s, fx * s + fy * c, fz])
        pos[2] = max(pos[2], FLOOR_Z + min_height)
    miss = np.linalg.norm(pipe[:, :2] - pos[:2], axis=1).min()
    return nav.done(), miss, step, nav.seen


if __name__ == "__main__":
    from pipe_detector.generate import SceneRandomizer
    from pipe_detector.detector import PipeDetector

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--no-water", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    gen, det, rng = SceneRandomizer(), PipeDetector(), np.random.default_rng(a.seed)
    ok = 0
    for ep in range(a.episodes):
        arrived, miss, steps, seen = run_episode(gen, det, rng, water=not a.no_water)
        success = arrived and miss < 0.5
        ok += success
        print(f"episode {ep:2d}: {'arrived' if arrived else 'timed out':9s} after {steps:3d} steps, "
              f"{miss:.2f} m from the pipe{'' if seen else ' (never saw it)'} -> {'OK' if success else 'FAIL'}",
              flush=True)
    print(f"reached the pipe in {ok}/{a.episodes} episodes")
    gen.sim.close()
