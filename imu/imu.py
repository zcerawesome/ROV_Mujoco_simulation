#!/usr/bin/env python3
"""
Test A - static bias / noise / drift characterization for the OAK-D Pro W BNO086 IMU.

HOW TO RUN IT
    Put the camera FLAT and DEAD STILL on a stable, level surface (a table that
    is not being touched). Do not bump it. Then:

        python3 imu_test_a.py --seconds 300              # 5 min collect + analyze
        python3 imu_test_a.py --from-csv imu_test_a.csv  # re-analyze an old log
        python3 imu_test_a.py --calibrate-mag            # calibrate the magnetometer first

WHAT IT MEASURES  (the baseline is physics - no external reference needed)
    A stationary body truly has:
        * angular rate      = 0 deg/s  on every axis   -> any gyro reading IS error
        * acceleration      = 1 g (9.81 m/s^2) straight down, 0 horizontal
        * orientation       = constant                 -> any heading change IS drift
    So we compare the sensor against those known truths:
        - gyro bias (deg/s)   = mean of each gyro axis   (should be 0)
        - gyro noise (deg/s)  = std  of each gyro axis    (1-sigma)
        - accel error         = |mean|g| - 9.81|         (should be 0)
        - raw-gyro yaw drift  = integral of gyro-Z over time (deg/min), both
                                as-measured and with the constant bias removed
        - Allan deviation     = separates white noise (ARW) from bias instability
        - fused heading drift = change in BNO086 rotation-vector yaw (deg/min)

    The fused-heading number is only trustworthy once the magnetometer is
    calibrated - run --calibrate-mag and watch the reported accuracy leave 180 deg.

Targets DepthAI v2 (XLinkOut / device.getOutputQueue API).
"""

import argparse
import csv
import math
import time

RAD2DEG = 180.0 / math.pi


def quat_to_yaw(i, j, k, w):
    """Yaw (rotation about Z), radians, from a unit quaternion (x=i, y=j, z=k, w=real)."""
    return math.atan2(2.0 * (w * k + i * j), 1.0 - 2.0 * (j * j + k * k))


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def collect(seconds, rate, rv_rate, csv_path):
    import depthai as dai

    gyro, accel, rv = [], [], []          # per-stream samples
    last_g = last_a = last_r = None       # dedupe by device timestamp
    t0 = None

    # Build the pipeline first, then hand it to the device, which starts it.
    pipeline = dai.Pipeline()
    imu = pipeline.create(dai.node.IMU)
    # Raw accel + gyro; fused rotation vector (9-axis, needs mag).
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, rate)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, rate)
    imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, rv_rate)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(20)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("imu")
    imu.out.link(xout.input)

    with dai.Device(pipeline) as device:
        name = device.getConnectedIMU()
        print(f"Connected IMU : {name}")
        try:
            print(f"IMU firmware  : {device.getIMUFirmwareVersion()}")
        except Exception:
            pass
        if "BNO" not in str(name).upper():
            print("WARNING: expected BNO086 (9-axis, fused). On a 6-axis BMI270 the "
                  "ROTATION_VECTOR / fused-heading section will be empty.")

        imuQueue = device.getOutputQueue(name="imu", maxSize=100, blocking=False)

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            # generic columns: for rv, (a,b,c,d,e) = (i, j, k, real, rvAccuracy_rad)
            w.writerow(["stream", "t_s", "a", "b", "c", "d", "e"])

            print(f"\nHold STILL. Logging {seconds:.0f} s ...", flush=True)
            start = time.monotonic()
            next_tick = start + 10.0
            while not device.isClosed() and time.monotonic() - start < seconds:
                data = imuQueue.get()
                for p in data.packets:
                    g = p.gyroscope
                    gts = g.getTimestamp().total_seconds()
                    if gts != last_g:
                        last_g = gts
                        t0 = gts if t0 is None else t0
                        t = gts - t0
                        gyro.append((t, g.x, g.y, g.z))
                        w.writerow(["gyro", f"{t:.6f}", g.x, g.y, g.z, "", ""])

                    a = p.acceleroMeter
                    ats = a.getTimestamp().total_seconds()
                    if ats != last_a:
                        last_a = ats
                        t0 = ats if t0 is None else t0
                        t = ats - t0
                        accel.append((t, a.x, a.y, a.z))
                        w.writerow(["accel", f"{t:.6f}", a.x, a.y, a.z, "", ""])

                    r = p.rotationVector
                    rts = r.getTimestamp().total_seconds()
                    if rts != last_r:
                        last_r = rts
                        t0 = rts if t0 is None else t0
                        t = rts - t0
                        rv.append((t, quat_to_yaw(r.i, r.j, r.k, r.real), r.rotationVectorAccuracy))
                        w.writerow(["rv", f"{t:.6f}", r.i, r.j, r.k, r.real, r.rotationVectorAccuracy])

                if time.monotonic() >= next_tick:
                    el = time.monotonic() - start
                    print(f"  {el:5.0f}s  gyro={len(gyro)}  accel={len(accel)}  rv={len(rv)}", flush=True)
                    next_tick += 10.0

    print(f"\nSaved raw log to {csv_path}")
    return gyro, accel, rv


def calibrate_mag(seconds, rv_rate):
    """Live magnetometer-calibration helper.

    The BNO086 calibrates its magnetometer opportunistically from motion: you
    have to show it every orientation. It reports how well that is going in two
    ways, both printed live here:
        * magnetometer accuracy   UNRELIABLE -> LOW -> MEDIUM -> HIGH
        * rotation-vector accuracy in degrees (180 = "I do not know my heading")
    Test A's fused-heading number only means anything once these leave the floor.
    """
    import depthai as dai

    pipeline = dai.Pipeline()
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, rv_rate)
    imu.enableIMUSensor(dai.IMUSensor.MAGNETOMETER_CALIBRATED, rv_rate)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(20)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("imu")
    imu.out.link(xout.input)

    print("\n" + "=" * 64)
    print("MAGNETOMETER CALIBRATION")
    print("=" * 64)
    print("Pick the camera up and, away from metal/monitors/motors:")
    print("  1. sweep it in slow figure-8s, wrist rotating as you go")
    print("  2. then rotate it fully about each of its three axes")
    print("Keep going until mag accuracy reads HIGH and RV accuracy drops well")
    print("below 180 deg. Ctrl-C to stop early.\n")

    with dai.Device(pipeline) as device:
        print(f"Connected IMU : {device.getConnectedIMU()}\n")
        q = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
        start = time.monotonic()
        next_print = 0.0
        best = None
        try:
            while not device.isClosed() and time.monotonic() - start < seconds:
                data = q.get()
                for p in data.packets:
                    r, m = p.rotationVector, p.magneticField
                    rv_deg = r.rotationVectorAccuracy * RAD2DEG
                    if best is None or rv_deg < best:
                        best = rv_deg
                    el = time.monotonic() - start
                    if el >= next_print:
                        print(f"  {el:5.0f}s  mag={m.accuracy.name:<10s} "
                              f"rv={r.accuracy.name:<10s} "
                              f"rv_accuracy={rv_deg:6.2f} deg   (best {best:6.2f})",
                              flush=True)
                        next_print = el + 1.0
        except KeyboardInterrupt:
            print("\n  stopped.")

    print("\n" + "=" * 64)
    if best is not None and best < 30.0:
        print(f"Calibration converged (best RV accuracy {best:.2f} deg).")
        print("Now re-run Test A - the fused-heading number is meaningful.")
    else:
        print(f"NOT converged (best RV accuracy {best if best is not None else float('nan'):.2f} deg).")
        print("If it never left 180 deg even with vigorous motion, the magnetometer")
        print("is not usable on this unit - treat the BNO086 as 6-axis and plan for")
        print("an external heading reference in the ROV.")
    print("=" * 64)


def load_csv(csv_path):
    gyro, accel, rv = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["t_s"])
            if row["stream"] == "gyro":
                gyro.append((t, float(row["a"]), float(row["b"]), float(row["c"])))
            elif row["stream"] == "accel":
                accel.append((t, float(row["a"]), float(row["b"]), float(row["c"])))
            elif row["stream"] == "rv":
                # a,b,c,d = quaternion i,j,k,real ; e = accuracy(rad)
                yaw = quat_to_yaw(float(row["a"]), float(row["b"]), float(row["c"]), float(row["d"]))
                rv.append((t, yaw, float(row["e"]) if row["e"] else float("nan")))
    return gyro, accel, rv


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def allan_dev(rate, dt, n_taus=40):
    """Overlapping Allan deviation of a rate signal sampled uniformly at `dt`.

    Returns (taus_s, sigma) with sigma in the same units as `rate`. The curve
    separates the two noise terms that a single std-dev conflates:
        slope -1/2  ->  white noise      (angle random walk)
        the minimum ->  bias instability (the floor you cannot average away)
    """
    import numpy as np

    x = np.asarray(rate, dtype=float)
    N = x.size
    theta = np.cumsum(x) * dt                    # integrated angle
    max_m = (N - 1) // 3                         # keep >= 3 clusters per tau
    if max_m < 1:
        return np.array([]), np.array([])
    ms = np.unique(np.floor(np.logspace(0, math.log10(max_m), n_taus)).astype(int))
    taus, sig = [], []
    for m in ms:
        if N - 2 * m < 1:
            continue
        d = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        var = (d ** 2).sum() / (2.0 * (m * dt) ** 2 * (N - 2 * m))
        taus.append(m * dt)
        sig.append(math.sqrt(var))
    return np.array(taus), np.array(sig)


def _allan_summary(name, taus, sig, dur_s):
    """Print ARW + bias-instability read off one Allan curve (units: deg/s in)."""
    import numpy as np

    if taus.size < 3:
        print(f"    {name}: too few samples for an Allan curve.")
        return
    # Angle random walk: sigma(tau) = ARW / sqrt(tau)  ->  read at tau = 1 s.
    i1 = int(np.argmin(np.abs(taus - 1.0)))
    arw = sig[i1] * math.sqrt(taus[i1]) * 60.0        # deg/s*sqrt(s) -> deg/sqrt(hr)
    # Bias instability: the minimum of the curve, / 0.664.
    imin = int(np.argmin(sig))
    bi = sig[imin] / 0.664
    flag = "" if taus[imin] < dur_s / 10.0 else "  (at tau near the edge - UNRELIABLE)"
    print(f"    {name}  ARW {arw:6.2f} deg/sqrt(hr)   "
          f"bias-instab {bi * 3600:7.1f} deg/hr @ tau={taus[imin]:.1f}s{flag}")


def analyze(gyro, accel, rv):
    import numpy as np

    print("\n" + "=" * 64)
    print("TEST A RESULTS  (baseline = a stationary body: 0 rot, 1 g down)")
    print("=" * 64)

    # ---- Gyro bias & noise -------------------------------------------------
    if len(gyro) > 2:
        G = np.asarray(gyro)                      # t, gx, gy, gz  (rad/s)
        t = G[:, 0]
        gd = G[:, 1:4] * RAD2DEG                  # deg/s
        bias, noise = gd.mean(axis=0), gd.std(axis=0)
        dur_min = (t[-1] - t[0]) / 60.0
        eff_hz = len(t) / (t[-1] - t[0]) if t[-1] > t[0] else 0.0
        print(f"\nGYRO  ({len(t)} samples, {eff_hz:.0f} Hz, {dur_min:.2f} min)")
        print(f"  bias  [truth 0]   x={bias[0]:+.4f}  y={bias[1]:+.4f}  z={bias[2]:+.4f}  deg/s")
        print(f"  noise (1-sigma)   x={noise[0]:.4f}  y={noise[1]:.4f}  z={noise[2]:.4f}  deg/s")
        print(f"  noise RMS         {np.sqrt((noise**2).sum()):.4f} deg/s")

        # Raw-gyro-integrated yaw drift (trapezoidal integral of gyro-Z),
        # shown both as-measured and with the constant bias subtracted - the
        # difference is what a one-line correction in your estimator buys.
        def _yaw_drift(gz_series):
            return np.concatenate(
                [[0.0], np.cumsum((gz_series[1:] + gz_series[:-1]) * 0.5 * np.diff(t))])[-1]

        gz = gd[:, 2]
        total = _yaw_drift(gz)
        total_c = _yaw_drift(gz - bias[2])          # bias-corrected
        print(f"\nRAW-GYRO YAW DRIFT  [truth: heading constant]")
        print(f"  as-measured       {total:+.2f} deg in {dur_min:.2f} min", end="")
        if dur_min > 0:
            print(f"   ({total / dur_min:+.2f} deg/min, {total / dur_min * 60:+.1f} deg/hr)")
        else:
            print()
        print(f"  bias-corrected    {total_c:+.2f} deg in {dur_min:.2f} min", end="")
        if dur_min > 0:
            print(f"   ({total_c / dur_min:+.2f} deg/min, {total_c / dur_min * 60:+.1f} deg/hr)")
        else:
            print()
        print(f"  -> the correction is just subtracting z-bias {bias[2]:+.4f} deg/s.")
        print("     CAUTION: the bias was measured ON THIS LOG, so the corrected number")
        print("              collapses to ~0 by construction. It is NOT a forecast. What")
        print("              you actually get on future data is the ARW/bias-instability")
        print("              below, and only after re-measuring bias at run temperature.")

        # ---- Allan deviation: separates white noise from bias instability ---
        dts = np.diff(t)
        dt = float(np.median(dts))
        jitter = float(dts.std())
        print(f"\nALLAN DEVIATION  (dt={dt * 1e3:.2f} ms, jitter {jitter * 1e3:.2f} ms)")
        if jitter > 0.5 * dt:
            print("  WARNING: sample spacing is too irregular for a trustworthy Allan curve.")
        dur_s = t[-1] - t[0]
        for ax in range(3):
            taus, sig = allan_dev(gd[:, ax], dt)
            _allan_summary("xyz"[ax], taus, sig, dur_s)
        print(f"  (tau beyond ~{dur_s / 10:.0f}s is unreliable on a {dur_s / 60:.1f} min log;")
        print("   run ~1 hr to see the true bias-instability floor.)")
    else:
        print("\nGYRO: not enough samples.")

    # ---- Accel vs gravity --------------------------------------------------
    if len(accel) > 2:
        A = np.asarray(accel)                     # t, ax, ay, az  (m/s^2)
        vec = A[:, 1:4]
        mag = np.linalg.norm(vec, axis=1)
        mean_vec = vec.mean(axis=0)
        print(f"\nACCEL  ({len(A)} samples)   [truth: |g| = 9.81, all in one axis]")
        print(f"  mean per axis     x={mean_vec[0]:+.3f}  y={mean_vec[1]:+.3f}  z={mean_vec[2]:+.3f}  m/s^2")
        print(f"  |g| magnitude     {mag.mean():.3f} m/s^2   (error {mag.mean() - 9.81:+.3f})")
        print(f"  |g| noise         {mag.std():.4f} m/s^2")
        # off-level tilt: angle between measured gravity and its nearest axis
        dominant = np.argmax(np.abs(mean_vec))
        horiz = math.sqrt(sum(mean_vec[k] ** 2 for k in range(3) if k != dominant))
        tilt = math.degrees(math.atan2(horiz, abs(mean_vec[dominant])))
        print(f"  off-level tilt    {tilt:.2f} deg  (gravity mostly on {'xyz'[dominant]}-axis; "
              f"0 = perfectly level for that mounting)")
    else:
        print("\nACCEL: not enough samples.")

    # ---- Fused heading drift (BNO086 rotation vector) ----------------------
    if len(rv) > 2:
        R = np.asarray(rv)                        # t, yaw(rad), acc(rad)
        t = R[:, 0]
        yaw = np.unwrap(R[:, 1]) * RAD2DEG
        dur_min = (t[-1] - t[0]) / 60.0
        fused = yaw[-1] - yaw[0]
        acc_deg = np.nanmean(R[:, 2]) * RAD2DEG
        print(f"\nFUSED HEADING (rotation vector, {len(t)} samples)  [truth: constant]")
        print(f"  heading drift     {fused:+.2f} deg over {dur_min:.2f} min", end="")
        if dur_min > 0:
            print(f"   ({fused / dur_min:+.2f} deg/min)")
        else:
            print()
        print(f"  reported accuracy {acc_deg:.2f} deg  (BNO086 self-estimate)")
        print("  NOTE: fused drift should be SMALL (mag-corrected) even when raw-gyro")
        print("        drift is large - that difference shows what fusion buys you.")
    else:
        print("\nFUSED HEADING: no rotation-vector samples (6-axis IMU, or RV disabled).")

    print("\n" + "=" * 64)
    print("Re-run this INSIDE the powered ROV (thrusters on, after mag calibration).")
    print("The bench-vs-ROV delta is the number that actually governs your autonomy.")
    print("=" * 64)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="OAK-D BNO086 static IMU characterization (Test A)")
    ap.add_argument("--seconds", type=float, default=300.0, help="collection duration (default 300 = 5 min)")
    ap.add_argument("--rate", type=int, default=200, help="raw accel/gyro report rate Hz (default 200)")
    ap.add_argument("--rv-rate", type=int, default=100, help="rotation-vector rate Hz (default 100)")
    ap.add_argument("--csv", default="imu_test_a.csv", help="output CSV path")
    ap.add_argument("--from-csv", metavar="PATH", help="skip collection; analyze an existing CSV")
    ap.add_argument("--calibrate-mag", action="store_true",
                    help="run the live magnetometer-calibration helper instead of Test A")
    args = ap.parse_args()

    if args.calibrate_mag:
        calibrate_mag(args.seconds, args.rv_rate)
        return

    if args.from_csv:
        gyro, accel, rv = load_csv(args.from_csv)
    else:
        gyro, accel, rv = collect(args.seconds, args.rate, args.rv_rate, args.csv)

    analyze(gyro, accel, rv)


if __name__ == "__main__":
    main()
