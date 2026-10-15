# ROV_Mujoco_simulation

MuJoCo simulation of the Purdue IEEE ROV, with a networked controller and
straight-line autonomy built on the depth controller from
[purduerov/X19-Core](https://github.com/purduerov/X19-Core) (`depth_control` branch).

## Setup

```bash
./install_requirements.sh        # pip deps; also needs g++ and python3-tk
```

## Running

```bash
./run.sh                         # straight-line autonomy, controller + sim
./run.sh --depth 2.0 --surge 0.6 --cruise-time 10
./run.sh --goal spin             # original OAK-D IMU spin (needs the camera)

python3 straight_line_test.py            # headless closed-loop test, all scenarios
python3 straight_line_test.py calm --video
python3 -m pytest tests/                 # same checks + core controller unit tests
```

Viewer keys: `1` free camera, `2` tracking, `3` front, `4` down, `q` quit.

## Straight-line autonomy

`behavior/straight_line.py` drives the ROV through three phases: **settle**
(dive to the target depth and lock the heading), **cruise** (surge forward
while holding depth and heading), then **stop** (keep holding). It only
uses what the real vehicle can measure: pressure depth and IMU attitude.

| piece | where | what |
|---|---|---|
| PID | `rov_core/controller.{hpp,cpp}` | X19-Core's `Controller`, vendored and compiled with g++ on first import, called via ctypes (`rov_core/core_controller.py`), so the sim runs the same C++ as the ROV |
| depth loop | `StraightLine.update` | the `depth_control.cpp` loop: `record_measurement(depth)` → `target` → `compute_filtered(dt)` |
| heading / roll / pitch | `StraightLine.update` | the same `Controller` class with its own gains; heading is unwrapped around its target so ±180° is handled |
| merge | `rov_core/thruster_control.py` | a Python version of the `thruster_control.hpp` stub `genThrusterCommand(measuredPose, joystickCommand, depthControlCommand)`. It rotates the world-frame depth command into the body frame through the measured attitude |
| mapping | `MuJoCoBase.set_wrench` | wrench → 8 thrusters (pseudo-inverse), compensated for weaker reverse thrust |

There is no position sensor, so "straight" means constant heading at
constant depth. A cross current still pushes the track sideways, just as it
would on the real vehicle (see the `cross_current` scenario).

### Results (`python3 straight_line_test.py`)

| scenario | depth err max | heading err max | cross-track max | distance |
|---|---|---|---|---|
| calm (noisy sensors, T200 asymmetry) | 0.4 cm | 0.8° | 0.1 cm | 9.1 m |
| 0.2 m/s head current | 0.4 cm | 0.8° | 0.1 cm | 7.2 m |
| 0.1 m/s cross current | 0.4 cm | 0.8° | 96 cm (crab, expected) | 9.1 m |
| start rolled 15°, turn across ±180° | 0.2 cm | 1.1° | 4.1 cm | 9.1 m |

Plots are in `results/`.

### Fixes to the X19-Core controller

The vendored copy differs from upstream in three places, marked `SIM FIX`.
`rov_core/upstream_fixes.patch` has the diff. With the upstream code
unmodified, the `calm` scenario fails: the ROV drives into the seabed and
flips over.

1. `double p = weights.k_p;` → `weights.k_p * error`. The P term ignored the error.
2. `last_error` starts at 0, so the first derivative was a full-scale kick. It is now seeded on the first sample.
3. The EMA filter starts at 0 m, so the controller first believes it is at the surface. It is now seeded with the first measurement.

Also in `depth_control.cpp` (not vendored): `controller->target` is never
set from `sub.getTargetDepth()`, and `dt` is hard-coded to 0.1 s.

## Simulation model

* **Gravity and buoyancy**: gravity was previously 0, so weight and
  buoyancy didn't exist. Gravity is now on, and buoyancy (MuJoCo has none) is
  applied at a centre of buoyancy 2 cm above the CoM. This gives a passive
  righting moment, slight positive buoyancy, and partial buoyancy at the surface (z = 0).
* **Hydrodynamic damping**: linear + quadratic, per body axis. MuJoCo's
  inertia-box drag let the vehicle reach 2.4 m/s surge and 18 rad/s yaw. It
  now reaches about 1.4 m/s surge, 1 m/s heave and 2.5 rad/s yaw. Coefficients
  are `<custom>` numerics in `models/rov.xml`.
* **Water current**: `MuJoCoBase(current=(vx, vy, vz))`. It acts through drag
  on the velocity relative to the water.
* **Thrusters**: T200-style reverse thrust is 78% of forward, and a 80 ms
  spin-up lag is modelled. `set_wrench` now accounts for the actuator gear,
  so it really takes N and N·m (it was off by 40×). It also scales the whole
  command on saturation instead of clipping per thruster.
* **Sensors**: a pressure-depth and IMU site, with noise.
  `read_telemetry()` returns them, and they ride back to `controller.py` with
  each frame.
* **Thrust mode**: `{"thrust": [x, y, z, roll, pitch, yaw]}` over the socket
  drives the thrusters. A bare list keeps the original velocity mode.
* **Misc**: the sim retries connecting to the controller, so `run.sh` has no
  start-order race. The renderer is created lazily, so headless runs need no
  GL. It falls back from EGL to GLFW when EGL is unavailable.
