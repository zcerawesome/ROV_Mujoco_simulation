#!/usr/bin/env python3
"""
Dead-reckoned pose tracker for the OAK-D Pro W BNO086 IMU.

Reports position and orientation relative to wherever the IMU was on its FIRST
sample: that sample defines the origin and the reference attitude, and
everything after is reported as a delta from it.

HOW TO RUN IT
    python3 track.py                          # live from the camera
    python3 track.py --seconds 60 --out t.csv # live, logged
    python3 track.py --from-csv imu_test_a.csv  # replay a Test A log

HOW TO USE IT FROM ANOTHER SCRIPT
    from imu import track
    pose = track.get_pose()          # live camera; first call starts the stream
    pose = track.get_pose("log.csv") # or replay a Test A CSV in real time
    pose.position    # np.array [x, y, z] m, relative to the first sample
    pose.rotation    # np.array [roll, pitch, yaw] deg, relative to the first sample
    pose.calibrated  # False (and zeros above) until the opening still window is done

    The first call spins up a daemon thread that reads the IMU, calibrates on
    the opening `calib_seconds` (hold still!), and dead-reckons from then on.
    Every later call is a non-blocking read of the latest pose.

    frame = track.get_frame()        # latest RGB frame from the same camera, or None
    frame.shape                      # (540, 960, 3) uint8, RGB (matches cv_processor)

    The OAK can only be opened by ONE process, so the RGB stream rides on the
    same pipeline as the IMU: get_frame() and get_pose() share one device
    connection and one background thread. get_frame() returns None until the
    first frame lands, and always None when replaying a CSV (no camera open).

READ THIS BEFORE TRUSTING THE POSITION
    Orientation is honest. It comes from the BNO086's fused rotation vector,
    which is bounded by the accelerometer (roll/pitch) and - once calibrated -
    the magnetometer (yaw). Errors stay small and do not accumulate.

    Position is NOT honest and cannot be made honest by this or any other
    script. It is the double integral of acceleration, so a constant residual
    bias `b` becomes a position error of 0.5*b*t^2 - it grows with the SQUARE
    of time and there is nothing in an IMU to bound it:

        residual bias        drift after 10 s      after 60 s
        0.01 m/s^2           0.5 m                 18 m
        0.05 m/s^2           2.5 m                 90 m
        0.10 m/s^2           5.0 m                 180 m

    Attitude error leaks gravity straight into that bias: being wrong about
    "down" by just 1 degree injects 9.81*sin(1deg) = 0.17 m/s^2.

    So: use the orientation. Treat the position as a decaying guess that is
    useful for a second or two, and fix it with an external reference (DVL,
    depth sensor, visual odometry, USBL). Run --from-csv on a log where the
    camera never moved to see exactly how fast it rots for YOUR unit.
"""

import argparse
import atexit
import csv
import math
import threading
import time
from typing import NamedTuple

import numpy as np

RAD2DEG = 180.0 / math.pi
G_NOMINAL = 9.80665


# --------------------------------------------------------------------------- #
# Quaternion helpers.  Convention: q = (w, x, y, z), rotates body -> world.
# --------------------------------------------------------------------------- #
def q_norm(q):
    n = math.sqrt(float(q @ q))
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_rot(q, v):
    """Rotate vector v by quaternion q."""
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def q_from_omega(w_body, dt):
    """Small-angle rotation quaternion from a body angular rate over dt."""
    theta = np.linalg.norm(w_body) * dt
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = w_body / np.linalg.norm(w_body)
    s = math.sin(theta / 2.0)
    return np.array([math.cos(theta / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def q_to_euler(q):
    """ZYX intrinsic euler angles (roll, pitch, yaw) in radians."""
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, s) if abs(s) >= 1.0 else math.asin(s)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def rv_to_q(i, j, k, real):
    return q_norm(np.array([real, i, j, k]))


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class DeadReckoner:
    """Tracks pose relative to the first sample.

    Orientation comes from the fused rotation vector (and, for comparison, from
    integrating the raw gyro). Position is the double integral of gravity-
    compensated acceleration, and is included with the caveats in the module
    docstring firmly in mind.
    """

    def __init__(self, q_ref, g_world, gyro_bias, zupt=False,
                 zupt_a=0.20, zupt_w=0.02):
        self.q_ref = q_ref                 # attitude at t=0, the reference
        self.g_world = g_world             # gravity in world frame, m/s^2
        self.gyro_bias = gyro_bias         # rad/s, body frame
        self.q_gyro = q_ref.copy()         # gyro-only attitude, for comparison
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.zupt = zupt
        self.zupt_a = zupt_a
        self.zupt_w = zupt_w
        self.zupt_hits = 0
        self.max_speed = 0.0

    def step(self, dt, a_body, w_body, q_fused):
        """Advance by dt. Returns (q_rel, a_world_linear, was_zupt)."""
        if dt <= 0 or dt > 1.0:            # skip gaps / bad timestamps
            return q_mul(q_conj(self.q_ref), q_fused), np.zeros(3), False

        w = w_body - self.gyro_bias
        self.q_gyro = q_norm(q_mul(self.q_gyro, q_from_omega(w, dt)))

        # Gravity-compensated acceleration in the world frame.
        a_w = q_rot(q_fused, a_body) - self.g_world

        still = False
        if self.zupt and np.linalg.norm(a_w) < self.zupt_a and np.linalg.norm(w) < self.zupt_w:
            self.vel[:] = 0.0              # zero-velocity update
            self.zupt_hits += 1
            still = True
        else:
            self.pos += self.vel * dt + 0.5 * a_w * dt * dt
            self.vel += a_w * dt

        self.max_speed = max(self.max_speed, float(np.linalg.norm(self.vel)))
        return q_mul(q_conj(self.q_ref), q_fused), a_w, still

    def rel_euler_deg(self, q_fused):
        r, p, y = q_to_euler(q_mul(q_conj(self.q_ref), q_fused))
        return r * RAD2DEG, p * RAD2DEG, y * RAD2DEG

    def gyro_euler_deg(self):
        r, p, y = q_to_euler(q_mul(q_conj(self.q_ref), self.q_gyro))
        return r * RAD2DEG, p * RAD2DEG, y * RAD2DEG


# --------------------------------------------------------------------------- #
def calibrate(samples, calib_s):
    """Estimate gyro bias and the world-frame gravity vector from the opening
    (assumed stationary) window. Returns (q_ref, g_world, gyro_bias, n_used)."""
    q_ref = rv_to_q(*samples[0][3])
    t0 = samples[0][0]
    win = [s for s in samples if s[0] - t0 <= calib_s] or [samples[0]]

    gyro_bias = np.mean([s[2] for s in win], axis=0)
    # Gravity as the sensor sees it while still, rotated into the world frame.
    g_world = np.mean([q_rot(rv_to_q(*s[3]), s[1]) for s in win], axis=0)
    return q_ref, g_world, gyro_bias, len(win)


class Pose(NamedTuple):
    """Snapshot of the dead-reckoned pose relative to the first sample."""
    position: np.ndarray    # [x, y, z] m, world frame
    rotation: np.ndarray    # [roll, pitch, yaw] deg
    calibrated: bool        # False while the opening still window is still filling


class PoseTracker:
    """Feeds raw IMU samples through calibration and a DeadReckoner, and hands
    out the latest pose on demand. Thread-safe: feed() may run in a background
    thread while pose() is polled from another.

    on_calibrated(q_ref, g_world, gyro_bias, n_used) and
    on_sample(elapsed_s, dr, q_fused, still) are optional hooks used by the CLI
    for reporting/logging."""

    def __init__(self, calib_seconds=2.0, zupt=False, on_calibrated=None, on_sample=None):
        self.calib_seconds = calib_seconds
        self.zupt = zupt
        self.on_calibrated = on_calibrated
        self.on_sample = on_sample
        self.dr = None
        self.elapsed = 0.0
        self._buf = []
        self._t0 = None
        self._tprev = None
        self._q = None
        self._lock = threading.Lock()

    def feed(self, t, a_body, w_body, rv):
        """Push one sample (t s, accel[3] m/s^2, gyro[3] rad/s, (i,j,k,real)).
        Buffers until calib_seconds have elapsed, then calibrates on that
        window, replays it through the estimator, and tracks live from there."""
        if self.dr is None:
            self._buf.append((t, a_body, w_body, rv))
            if self._buf[-1][0] - self._buf[0][0] < self.calib_seconds:
                return
            q_ref, g_world, gyro_bias, n = calibrate(self._buf, self.calib_seconds)
            if self.on_calibrated:
                self.on_calibrated(q_ref, g_world, gyro_bias, n)
            dr = DeadReckoner(q_ref, g_world, gyro_bias, zupt=self.zupt)
            buf, self._buf = self._buf, []
            for s in buf:
                self._step(dr, *s)
            with self._lock:
                self.dr = dr
            return
        self._step(self.dr, t, a_body, w_body, rv)

    def _step(self, dr, t, a_body, w_body, rv):
        if self._t0 is None:
            self._t0 = t
        dt = 0.0 if self._tprev is None else t - self._tprev
        self._tprev = t
        q = rv_to_q(*rv)
        with self._lock:
            _, _, still = dr.step(dt, a_body, w_body, q)
            self._q = q
            self.elapsed = t - self._t0
        if self.on_sample:
            self.on_sample(self.elapsed, dr, q, still)

    def pose(self):
        """Latest Pose. Zeros with calibrated=False until calibration is done."""
        with self._lock:
            if self.dr is None:
                return Pose(np.zeros(3), np.zeros(3), False)
            return Pose(self.dr.pos.copy(),
                        np.array(self.dr.rel_euler_deg(self._q)), True)


def report_header(q_ref, g_world, gyro_bias, n_calib, calib_s, zupt):
    print("=" * 76)
    print("DEAD-RECKONED POSE  (origin + reference attitude = first sample)")
    print("=" * 76)
    print(f"  calibration window  {calib_s:.1f} s ({n_calib} samples, assumed stationary)")
    print(f"  gyro bias           {gyro_bias * RAD2DEG} deg/s")
    print(f"  gravity (world)     [{g_world[0]:+.3f} {g_world[1]:+.3f} {g_world[2]:+.3f}]"
          f"  |g|={np.linalg.norm(g_world):.3f} m/s^2")
    err = np.linalg.norm(g_world) - G_NOMINAL
    print(f"  |g| error vs 9.807  {err:+.3f} m/s^2", end="")
    if abs(err) > 0.05:
        print(f"   <-- projects to {0.5 * abs(err) * 100:.0f} m of position error at t=10 s")
    else:
        print()
    print(f"  ZUPT                {'on' if zupt else 'off'}")
    print("-" * 76)
    print("     t     |        position (m)        |   attitude vs start (deg)   | speed")
    print("    (s)    |     x       y       z      |  roll    pitch    yaw       | (m/s)")
    print("-" * 76)


def report_line(t, dr, q_fused, still):
    r, p, y = dr.rel_euler_deg(q_fused)
    print(f"  {t:7.2f}  | {dr.pos[0]:+7.2f} {dr.pos[1]:+7.2f} {dr.pos[2]:+7.2f}    "
          f"| {r:+7.2f} {p:+7.2f} {y:+7.2f}    | {np.linalg.norm(dr.vel):5.2f}"
          f"{'  [still]' if still else ''}", flush=True)


def report_footer(dr, t_end, stationary_truth):
    print("-" * 76)
    dist = float(np.linalg.norm(dr.pos))
    print(f"  final position      [{dr.pos[0]:+.2f} {dr.pos[1]:+.2f} {dr.pos[2]:+.2f}] m"
          f"   |p| = {dist:.2f} m")
    print(f"  final speed         {np.linalg.norm(dr.vel):.2f} m/s"
          f"   (peak {dr.max_speed:.2f} m/s)")
    if dr.zupt:
        print(f"  ZUPT applied on     {dr.zupt_hits} samples")
    if stationary_truth:
        print()
        print("  THE CAMERA NEVER MOVED, so true position is [0 0 0] and true speed is 0.")
        print(f"  Everything above is pure integration error: {dist:.2f} m after {t_end:.0f} s.")
        if t_end > 0:
            print(f"  That is an effective residual bias of ~{2 * dist / t_end ** 2:.4f} m/s^2.")
    print("=" * 76)


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def samples_from_csv(path):
    """Replay a Test A log. Yields (t, accel[3], gyro[3], (i,j,k,real))."""
    acc, gyr, rot = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["t_s"])
            s = row["stream"]
            if s == "accel":
                acc.append((t, float(row["a"]), float(row["b"]), float(row["c"])))
            elif s == "gyro":
                gyr.append((t, float(row["a"]), float(row["b"]), float(row["c"])))
            elif s == "rv":
                rot.append((t, float(row["a"]), float(row["b"]),
                            float(row["c"]), float(row["d"])))
    if not acc or not rot:
        raise SystemExit(f"{path}: needs both accel and rv rows to track pose.")

    A = np.asarray(acc)
    G = np.asarray(gyr) if gyr else None
    R = np.asarray(rot)

    # Accel drives the clock; nearest-neighbour match gyro and rv onto it.
    def nearest(src, t):
        i = int(np.clip(np.searchsorted(src[:, 0], t), 1, len(src) - 1))
        return i if abs(src[i, 0] - t) < abs(src[i - 1, 0] - t) else i - 1

    out = []
    for k in range(len(A)):
        t = A[k, 0]
        ri = nearest(R, t)
        gi = nearest(G, t) if G is not None else None
        out.append((t, A[k, 1:4],
                    G[gi, 1:4] if G is not None else np.zeros(3),
                    tuple(R[ri, 1:5])))
    return out


def replay_csv(path, on_sample, realtime=True, stop=None):
    """Feed a Test A log through on_sample(t, accel, gyro, rv). With realtime
    the samples are paced by their timestamps so it behaves like the camera."""
    samples = samples_from_csv(path)
    wall0, t0 = time.monotonic(), samples[0][0]
    for t, a, w, rv in samples:
        if stop is not None and stop.is_set():
            return
        if realtime:
            lag = (t - t0) - (time.monotonic() - wall0)
            if lag > 0:
                time.sleep(lag)
        on_sample(t, a, w, rv)


RGB_FPS = 30
RGB_ISP_SCALE = (1, 2)      # 1080p * 1/2 = 960x540, the size cv_processor expects


def samples_from_device(seconds, rate, rv_rate, on_sample, stop=None, on_frame=None):
    """Stream live from the camera, calling on_sample(t, accel, gyro, rv).
    Runs for `seconds` (math.inf = forever) or until `stop` (an Event) is set.

    If on_frame is given, the RGB camera is added to the same pipeline and
    on_frame(rgb_ndarray) is called with each new 960x540 frame. It shares the
    device with the IMU because the OAK can only be opened by one process."""
    import depthai as dai

    pipeline = dai.Pipeline()
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, rate)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, rate)
    imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, rv_rate)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(20)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("imu")
    imu.out.link(xout.input)

    if on_frame is not None:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setIspScale(*RGB_ISP_SCALE)
        cam.setFps(RGB_FPS)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam.isp.link(xout_rgb.input)

    with dai.Device(pipeline) as device:
        print(f"Connected IMU : {device.getConnectedIMU()}")
        q = device.getOutputQueue(name="imu", maxSize=100, blocking=False)
        # maxSize=1 + non-blocking: the device drops stale frames rather than
        # queueing them, so a slow consumer always sees the newest image.
        q_rgb = (device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
                 if on_frame is not None else None)
        start = time.monotonic()
        last = None
        try:
            while (not device.isClosed() and time.monotonic() - start < seconds
                   and not (stop is not None and stop.is_set())):
                if q_rgb is not None:
                    img = q_rgb.tryGet()
                    if img is not None:
                        # getCvFrame() is BGR; the rest of the project (cv_processor,
                        # controller.recv_frame) works in RGB.
                        on_frame(img.getCvFrame()[:, :, ::-1].copy())
                for p in q.get().packets:
                    a, g, r = p.acceleroMeter, p.gyroscope, p.rotationVector
                    t = a.getTimestamp().total_seconds()
                    if t == last:
                        continue
                    last = t
                    on_sample(t,
                              np.array([a.x, a.y, a.z]),
                              np.array([g.x, g.y, g.z]),
                              (r.i, r.j, r.k, r.real))
        except KeyboardInterrupt:
            print("\n  stopped.")


# --------------------------------------------------------------------------- #
# One-call API
# --------------------------------------------------------------------------- #
def start_background(source=None, calib_seconds=2.0, zupt=False, rate=200, rv_rate=100,
                     camera=True):
    """Start reading the IMU in a daemon thread and return the PoseTracker it
    feeds. source=None streams from the camera; a path replays that CSV in
    real time. The thread is stopped automatically at interpreter exit.

    With camera=True (and a live source) the RGB stream is opened on the same
    device and the latest frame is kept for get_frame()."""
    tracker = PoseTracker(calib_seconds=calib_seconds, zupt=zupt)
    stop = threading.Event()

    def run():
        if source is None:
            print(f"IMU: hold STILL for the first {calib_seconds:.0f} s (gravity/bias calibration)...")
            samples_from_device(math.inf, rate, rv_rate, tracker.feed, stop=stop,
                                on_frame=_store_frame if camera else None)
        else:
            replay_csv(source, tracker.feed, realtime=True, stop=stop)

    thread = threading.Thread(target=run, name="imu-track", daemon=True)
    thread.start()

    def shutdown():
        stop.set()
        thread.join(timeout=2.0)
    atexit.register(shutdown)
    tracker.stop = shutdown
    return tracker


_tracker = None
_tracker_lock = threading.Lock()

_frame = None
_frame_lock = threading.Lock()


def _store_frame(rgb):
    """Frame sink for samples_from_device: keep only the newest RGB frame."""
    global _frame
    with _frame_lock:
        _frame = rgb


def _ensure_started(source, calib_seconds, zupt):
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = start_background(source, calib_seconds=calib_seconds, zupt=zupt)
        return _tracker


def get_frame(source=None, calib_seconds=2.0, zupt=False):
    """Latest RGB frame from the OAK's colour camera, as a (540, 960, 3) uint8
    ndarray in RGB order (the convention cv_processor uses), or None if no frame
    has arrived yet.

    Shares the device connection and background thread with get_pose(): the
    first call to either starts the stream, and the arguments only matter on
    that first call. Returns None for the lifetime of a CSV replay, since no
    camera is opened then. Each call returns the same array object until a new
    frame lands, so copy it before modifying in place."""
    _ensure_started(source, calib_seconds, zupt)
    with _frame_lock:
        return _frame


def get_pose(source=None, calib_seconds=2.0, zupt=False):
    """Current position and orientation from the IMU, as a Pose.

    The first call starts the IMU stream (see start_background); `source` and
    the calibration options only matter on that first call. Every call returns
    immediately with the latest estimate: Pose(position[3] m, rotation
    [roll, pitch, yaw] deg, calibrated). Both arrays are zero and
    calibrated=False until the opening still window has been processed.
    Remember the module docstring: trust rotation, distrust position."""
    return _ensure_started(source, calib_seconds, zupt).pose()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Track IMU position + rotation relative to the first sample")
    ap.add_argument("--from-csv", metavar="PATH", help="replay a Test A CSV instead of the camera")
    ap.add_argument("--seconds", type=float, default=60.0, help="live capture duration (default 60)")
    ap.add_argument("--rate", type=int, default=200, help="raw accel/gyro rate Hz (default 200)")
    ap.add_argument("--rv-rate", type=int, default=100, help="rotation-vector rate Hz (default 100)")
    ap.add_argument("--calib-seconds", type=float, default=2.0,
                    help="opening stationary window used for gravity/bias (default 2; 0 = first sample only)")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between printed lines")
    ap.add_argument("--zupt", action="store_true",
                    help="zero velocity whenever the IMU looks still (curbs position blow-up)")
    ap.add_argument("--stationary", action="store_true",
                    help="assert the IMU never moved, so the final position IS the error")
    ap.add_argument("--out", metavar="PATH", help="write the pose track to CSV")
    args = ap.parse_args()

    writer = fh = None
    if args.out:
        fh = open(args.out, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["t_s", "x_m", "y_m", "z_m", "vx", "vy", "vz",
                         "roll_deg", "pitch_deg", "yaw_deg",
                         "gyro_roll_deg", "gyro_pitch_deg", "gyro_yaw_deg"])

    state = {"next": 0.0}

    def on_calibrated(q_ref, g_world, gyro_bias, n):
        report_header(q_ref, g_world, gyro_bias, n, args.calib_seconds, args.zupt)

    def on_sample(el, dr, q, still):
        if writer:
            r, p, y = dr.rel_euler_deg(q)
            gr, gp, gy = dr.gyro_euler_deg()
            writer.writerow([f"{el:.6f}", *(f"{v:.6f}" for v in dr.pos),
                             *(f"{v:.6f}" for v in dr.vel),
                             f"{r:.4f}", f"{p:.4f}", f"{y:.4f}",
                             f"{gr:.4f}", f"{gp:.4f}", f"{gy:.4f}"])
        if el >= state["next"]:
            report_line(el, dr, q, still)
            state["next"] = el + args.interval

    tracker = PoseTracker(calib_seconds=args.calib_seconds, zupt=args.zupt,
                          on_calibrated=on_calibrated, on_sample=on_sample)

    if args.from_csv:
        replay_csv(args.from_csv, tracker.feed, realtime=False)
    else:
        print(f"Hold STILL for the first {args.calib_seconds:.0f} s (gravity/bias calibration)...")
        samples_from_device(args.seconds, args.rate, args.rv_rate, tracker.feed)

    if tracker.dr is None:
        raise SystemExit("no samples captured.")
    report_footer(tracker.dr, tracker.elapsed, args.stationary or bool(args.from_csv))
    if fh:
        fh.close()
        print(f"  track written to {args.out}")


if __name__ == "__main__":
    main()
