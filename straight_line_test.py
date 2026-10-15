"""Closed-loop test of the straight-line autonomy in the MuJoCo sim.

Runs the ROV headless (no window, no GPU needed) through a set of scenarios,
scores each against ground truth, and writes plots to results/.

    python3 straight_line_test.py                 # all scenarios
    python3 straight_line_test.py calm --video    # one scenario, plus an mp4
    python3 -m pytest tests/                      # the same checks, as tests

The autonomy only ever sees the sim's sensor channels (noisy depth and IMU);
the "truth" channel is used here for scoring and nothing else.
"""
import argparse
import math
import os
import sys
from dataclasses import dataclass, field

from simulation import MuJoCoBase   # first: picks the GL backend before mujoco loads
import mujoco as mj
import numpy as np

from behavior.straight_line import StraightLine, wrap_angle
from rov_core.thruster_control import quat_to_rpy

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "models", "rov.xml")
RESULTS_DIR = os.path.join(HERE, "results")
CONTROL_DT = 1 / 60          # same exchange rate as run_networked()


@dataclass
class Scenario:
    name: str
    description: str
    current: tuple = (0.0, 0.0, 0.0)
    target_depth: float = 1.5
    surge: float = 0.5
    cruise_time: float = 8.0
    start_yaw_deg: float = 0.0
    start_roll_deg: float = 0.0
    heading_deg: float = None          # None: lock the heading at start
    # pass/fail limits, during cruise
    max_depth_error: float = 0.10      # m
    max_heading_error_deg: float = 3.0
    max_cross_track: float = None      # m; None when a current makes drift expected
    min_distance: float = 5.0          # m along the line


SCENARIOS = {s.name: s for s in [
    Scenario("calm", "still water, noisy sensors, T200 thrust asymmetry",
             max_cross_track=0.25),
    Scenario("head_current", "0.2 m/s current against the direction of travel",
             current=(-0.2, 0.0, 0.0), min_distance=3.5, max_cross_track=0.25),
    Scenario("cross_current", "0.1 m/s current across the track: heading holds, track crabs",
             current=(0.0, 0.1, 0.0)),
    Scenario("wrap_and_tilt", "starts rolled 15 deg facing -170 deg, ordered to 170 deg "
             "(the short way is across +-180)", start_yaw_deg=-170.0, start_roll_deg=15.0,
             heading_deg=170.0, target_depth=2.0, max_cross_track=0.25),
]}


@dataclass
class Result:
    scenario: Scenario
    log: dict
    metrics: dict
    failures: list = field(default_factory=list)

    @property
    def passed(self):
        return not self.failures


def _set_start_pose(sim, scenario):
    yaw, roll = math.radians(scenario.start_yaw_deg), math.radians(scenario.start_roll_deg)
    qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    qx = np.array([math.cos(roll / 2), math.sin(roll / 2), 0, 0])
    q = np.zeros(4)
    mj.mju_mulQuat(q, qz, qx)
    sim.data.qpos[3:7] = q
    mj.mj_forward(sim.model, sim.data)


def run_scenario(scenario, duration=None, video_path=None, seed=0):
    sim = MuJoCoBase(XML_PATH, current=scenario.current, seed=seed,
                     width=640, height=360)
    _set_start_pose(sim, scenario)
    sim.command_thrust(np.zeros(6))
    heading = None if scenario.heading_deg is None else math.radians(scenario.heading_deg)
    behavior = StraightLine(sim.axis_max, target_depth=scenario.target_depth,
                            surge=scenario.surge, cruise_time=scenario.cruise_time,
                            heading=heading)
    duration = duration or (scenario.cruise_time + 30.0)
    # the depth target is for the pressure sensor, which sits above the CoM
    sensor_site = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_SITE, "imu")

    keys = ("t", "phase", "pos", "depth_true", "depth_meas", "rpy_true", "cmd", "ctrl")
    log = {k: [] for k in keys}
    frames, next_frame = [], 0.0
    while sim.data.time < duration:
        tel = sim.read_telemetry()
        cmd = behavior.update(tel, CONTROL_DT)
        sim.command_thrust(cmd)
        target = sim.data.time + CONTROL_DT
        while sim.data.time < target:
            sim.step()

        log["t"].append(sim.data.time)
        log["phase"].append(behavior.phase)
        log["pos"].append(sim.data.qpos[0:3].copy())
        log["depth_true"].append(-sim.data.site_xpos[sensor_site, 2])
        log["depth_meas"].append(tel["depth"])
        log["rpy_true"].append(quat_to_rpy(sim.data.qpos[3:7]))
        log["cmd"].append(np.array(cmd))
        log["ctrl"].append(sim.data.ctrl.copy())
        if video_path and sim.data.time >= next_frame:
            frames.append(sim._frame("tracking"))
            next_frame += 1 / 30
        if behavior.done() and behavior._phase_time > 5.0:
            break

    log = {k: (np.array(v) if k != "phase" else np.array(v, dtype=object)) for k, v in log.items()}
    if video_path:
        import mediapy as media
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        media.write_video(video_path, frames, fps=30)
        print(f"  video: {video_path}")
    sim.close()
    return score(scenario, log, behavior.heading_target)


def score(scenario, log, heading_target):
    cruise = log["phase"] == "cruise"
    m = {}
    if not cruise.any():
        return Result(scenario, log, m, ["never reached the cruise phase (depth never settled)"])

    i0 = np.argmax(cruise)
    settle_time = log["t"][i0]
    pos = log["pos"][cruise]
    start = pos[0, :2]
    line = np.array([math.cos(heading_target), math.sin(heading_target)])
    rel = pos[:, :2] - start
    along = rel @ line
    cross = rel @ np.array([-line[1], line[0]])

    depth_err = np.abs(log["depth_true"][cruise] - scenario.target_depth)
    heading_err = np.degrees(np.abs(wrap_angle(log["rpy_true"][cruise, 2] - heading_target)))
    tilt = np.degrees(np.abs(log["rpy_true"][cruise, :2])).max()
    m.update(settle_time=settle_time, distance=along[-1],
             speed=along[-1] / (log["t"][cruise][-1] - settle_time),
             depth_err_max=depth_err.max(), depth_err_rms=np.sqrt(np.mean(depth_err ** 2)),
             heading_err_max=heading_err.max(), cross_track_max=np.abs(cross).max(),
             cross_track_end=cross[-1], tilt_max=tilt)

    f = []
    if m["depth_err_max"] > scenario.max_depth_error:
        f.append(f"depth error {m['depth_err_max']:.3f} m > {scenario.max_depth_error} m")
    if m["heading_err_max"] > scenario.max_heading_error_deg:
        f.append(f"heading error {m['heading_err_max']:.2f} deg > {scenario.max_heading_error_deg} deg")
    if scenario.max_cross_track is not None and m["cross_track_max"] > scenario.max_cross_track:
        f.append(f"cross-track {m['cross_track_max']:.3f} m > {scenario.max_cross_track} m")
    if m["distance"] < scenario.min_distance:
        f.append(f"distance {m['distance']:.2f} m < {scenario.min_distance} m")
    return Result(scenario, log, m, f)


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log, s = result.log, result.scenario
    t = log["t"]
    cruise = log["phase"] == "cruise"
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{s.name}: {s.description}  [{'PASS' if result.passed else 'FAIL'}]")

    a = ax[0, 0]
    a.plot(log["pos"][:, 0], log["pos"][:, 1], color="0.6", label="settle / stop")
    a.plot(log["pos"][cruise, 0], log["pos"][cruise, 1], color="C0", lw=2, label="cruise")
    a.set_title("track, top-down (m)"); a.set_xlabel("x"); a.set_ylabel("y")
    a.axis("equal"); a.legend(); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(t, log["depth_meas"], color="0.75", lw=0.8, label="pressure sensor")
    a.plot(t, log["depth_true"], color="C0", label="true")
    a.axhline(s.target_depth, color="C3", ls="--", label="target")
    a.invert_yaxis(); a.set_title("depth (m)"); a.set_xlabel("t (s)"); a.legend(); a.grid(alpha=0.3)

    a = ax[1, 0]
    for k, name in enumerate(("roll", "pitch", "yaw")):
        a.plot(t, np.degrees(log["rpy_true"][:, k]), label=name)
    a.set_title("attitude, true (deg)"); a.set_xlabel("t (s)"); a.legend(); a.grid(alpha=0.3)

    a = ax[1, 1]
    for k, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        a.plot(t, log["cmd"][:, k], label=name, lw=0.9)
    a.set_title("normalized body command"); a.set_xlabel("t (s)"); a.legend(ncol=3); a.grid(alpha=0.3)
    for a in (ax[0, 1], ax[1, 0], ax[1, 1]):
        if cruise.any():
            a.axvspan(t[cruise][0], t[cruise][-1], color="C2", alpha=0.08)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="*", choices=[[], *SCENARIOS], default=[],
                    help="which to run (default: all)")
    ap.add_argument("--video", action="store_true", help="also record results/<name>.mp4")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    names = args.scenarios or list(SCENARIOS)
    ok = True
    for name in names:
        s = SCENARIOS[name]
        print(f"[{name}] {s.description}")
        video = os.path.join(RESULTS_DIR, f"{name}.mp4") if args.video else None
        r = run_scenario(s, video_path=video)
        m = r.metrics
        if m:
            print(f"  settled in {m['settle_time']:.1f} s, cruised {m['distance']:.2f} m at {m['speed']:.2f} m/s")
            print(f"  depth error  max {m['depth_err_max'] * 100:.1f} cm, rms {m['depth_err_rms'] * 100:.1f} cm")
            print(f"  heading err  max {m['heading_err_max']:.2f} deg   roll/pitch max {m['tilt_max']:.2f} deg")
            print(f"  cross-track  max {m['cross_track_max'] * 100:.1f} cm, at end {m['cross_track_end'] * 100:+.1f} cm")
        for f in r.failures:
            print(f"  FAIL: {f}")
        print("  PASS" if r.passed else "  FAILED")
        if not args.no_plot:
            path = os.path.join(RESULTS_DIR, f"{name}.png")
            plot(r, path)
            print(f"  plot: {path}")
        ok &= r.passed
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
