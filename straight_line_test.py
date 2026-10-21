"""Closed-loop test of X19-Core's straight-line autonomy in the MuJoCo sim.

For each scenario this starts the Core nodes, built from the `core`
submodule (depth_control, thruster_control, straight_line), and runs the
simulated ROV on the same ZeroMQ bus (sim_bridge.py). Everything runs in real
time, headless. Each run is scored against the sim's ground truth, and plots
go to results/.

    python3 straight_line_test.py                 # all scenarios (~1.5 min)
    python3 straight_line_test.py calm --video    # one scenario, plus an mp4
    python3 -m pytest tests/                      # the same checks, as tests

The Core nodes only see what the bridge publishes (noisy depth and IMU); the
sim's ground truth is used here for scoring and nothing else.
"""
import argparse
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

from sim_bridge import SimBridge, XML_PATH, CORE   # first: picks the GL backend before mujoco loads
from simulation import MuJoCoBase, quat_to_rpy
import mujoco as mj
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CORE_BUILD = os.path.join(CORE, "build")
CORE_NODES = ("depth_control", "thruster_control", "straight_line")
LOG_DT = 1 / 60
STOP_HOLD = 4.0              # s of the stop phase to record before ending


def wrap_angle(a):
    """Wrap to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def build_core():
    """Configure (once) and build the Core nodes. Needs cmake, protobuf,
    cppzmq and Eigen; set CMAKE_PREFIX_PATH if Eigen isn't installed system-wide."""
    if not os.path.exists(os.path.join(CORE, "CMakeLists.txt")):
        raise RuntimeError("core/ is empty: run `git submodule update --init`")
    if not os.path.exists(os.path.join(CORE_BUILD, "CMakeCache.txt")):
        subprocess.run(["cmake", "-S", CORE, "-B", CORE_BUILD, "-DCMAKE_BUILD_TYPE=Release"],
                       check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", CORE_BUILD, "-j", "--target", *CORE_NODES],
                   check=True, stdout=subprocess.DEVNULL)


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
    node_args: tuple = ()              # extra args for the straight_line node


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


def _start_core_nodes(scenario, log_dir):
    args = ["--depth", str(scenario.target_depth), "--surge", str(scenario.surge),
            "--cruise-time", str(scenario.cruise_time), "--exit-after", str(STOP_HOLD + 2)]
    if scenario.heading_deg is not None:
        args += ["--heading-deg", str(scenario.heading_deg)]
    args += list(scenario.node_args)
    os.makedirs(log_dir, exist_ok=True)
    procs = []
    for name in CORE_NODES:
        cmd = [os.path.join(CORE_BUILD, name)] + (args if name == "straight_line" else [])
        out = open(os.path.join(log_dir, f"{scenario.name}.{name}.log"), "w")
        procs.append((subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT), out))
    return procs


def _stop_core_nodes(procs):
    for p, _ in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    for p, out in procs:
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
        out.close()


def run_scenario(scenario, duration=None, video_path=None, seed=0):
    build_core()
    sim = MuJoCoBase(XML_PATH, current=scenario.current, seed=seed,
                     width=640, height=360)
    _set_start_pose(sim, scenario)
    bridge = SimBridge(sim, listen_status=True)
    procs = _start_core_nodes(scenario, os.path.join(RESULTS_DIR, "logs"))
    duration = duration or (scenario.cruise_time + 40.0)
    # the depth target is for the pressure sensor, which sits above the CoM
    sensor_site = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_SITE, "imu")

    keys = ("t", "phase", "pos", "depth_true", "rpy_true", "cmd", "ctrl")
    log = {k: [] for k in keys}
    frames = []
    heading_target = [None]
    stop_since = [None]

    def on_tick(s):
        status = bridge.status
        phase = status.phase if status is not None else "wait"
        if status is not None:
            heading_target[0] = status.target_heading
        log["t"].append(s.data.time)
        log["phase"].append(phase)
        log["pos"].append(s.data.qpos[0:3].copy())
        log["depth_true"].append(-s.data.site_xpos[sensor_site, 2])
        log["rpy_true"].append(quat_to_rpy(s.data.qpos[3:7]))
        log["cmd"].append(bridge.command.copy())
        log["ctrl"].append(s.data.ctrl.copy())
        if video_path and len(log["t"]) % 2 == 0:          # 30 fps
            frames.append(s._frame("tracking"))
        if phase == "stop":
            stop_since[0] = stop_since[0] if stop_since[0] is not None else s.data.time
            return s.data.time - stop_since[0] >= STOP_HOLD
        return False

    try:
        bridge.run_realtime(duration=duration, on_tick=on_tick, tick_dt=LOG_DT)
    finally:
        _stop_core_nodes(procs)
        bridge.close()

    log = {k: (np.array(v) if k != "phase" else np.array(v, dtype=object)) for k, v in log.items()}
    if video_path:
        import mediapy as media
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        media.write_video(video_path, frames, fps=30)
        print(f"  video: {video_path}")
    sim.close()
    if heading_target[0] is None:
        return Result(scenario, log, {}, ["no AutonomyStatus from straight_line (did the Core nodes start?)"])
    return score(scenario, log, heading_target[0])


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
    a.plot(t, log["depth_true"], color="C0", label="true (at the pressure sensor)")
    a.axhline(s.target_depth, color="C3", ls="--", label="target")
    a.invert_yaxis(); a.set_title("depth (m)"); a.set_xlabel("t (s)"); a.legend(); a.grid(alpha=0.3)

    a = ax[1, 0]
    for k, name in enumerate(("roll", "pitch", "yaw")):
        a.plot(t, np.degrees(log["rpy_true"][:, k]), label=name)
    a.set_title("attitude, true (deg)"); a.set_xlabel("t (s)"); a.legend(); a.grid(alpha=0.3)

    a = ax[1, 1]
    for k, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        a.plot(t, log["cmd"][:, k], label=name, lw=0.9)
    a.set_title("ThrusterCommand from thruster_control"); a.set_xlabel("t (s)"); a.legend(ncol=3); a.grid(alpha=0.3)
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
