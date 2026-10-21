# ROV_Mujoco_simulation

MuJoCo simulation of the Purdue IEEE ROV, with a networked controller and
straight-line autonomy built on the depth controller from
[purduerov/X19-Core](https://github.com/purduerov/X19-Core) (`depth_control` branch).

## Setup

```bash
git clone --recurse-submodules https://github.com/zcerawesome/ROV_Mujoco_simulation
./install_requirements.sh   # pip deps, then builds the X19-Core nodes in core/build
```

The Core nodes need `protobuf-compiler libprotobuf-dev libzmq3-dev cppzmq-dev
libeigen3-dev cmake g++`, the same packages as X19-Core's `scripts/setup.sh`
plus Eigen. The viewer also needs `python3-tk`.

## Running

```bash
./run.sh                                       # straight line: 1.5 m, surge 0.5, 8 s
./run.sh --depth 2 --surge 0.6 --cruise-time 10 --heading-deg 90
SIM_ARGS="--current 0 0.1 0" ./run.sh          # with a cross current
SIM_ARGS="--headless" ./run.sh                 # no window

python3 straight_line_test.py                  # closed-loop scenarios, headless (~1.5 min)
python3 straight_line_test.py calm --video
python3 -m pytest tests/

./run_tcp.sh                                   # original controller.py/Main.py OAK-D spin
```

Viewer keys: `1` free camera, `2` tracking, `3` front, `4` down, `space` pause, `q` quit.

## Straight-line autonomy on X19-Core

The sim doesn't contain its own control code. `core/` is a submodule of
[X19-Core](https://github.com/adarsh-siva/X19-Core/tree/sim-straight-line)
(fork, branch `sim-straight-line`, built on the team's `depth_control`
branch). `run.sh` starts its C++ nodes with Core's own `scripts/launch.py`,
and they talk to the simulated ROV over ZeroMQ + protobuf, exactly as they
would on the Pi:

```
 straight_line ──depthControlCommand :5560──▶ depth_control ──depthControlOutput :5561──┐
      │                                            ▲                                    ▼
      └──────────ControllerCommand :5562──────────────────────────────────────▶ thruster_control
                                                   │                       (genThrusterCommand)
      ▲          SensorData  telemetry :5555 ──────┘                                    │
      └───────── ImuData     imu       :5557 ◀── sim_bridge.py ◀── ThrusterCommand :5563┘
                                                (MuJoCo ROV)
```

* `sim_bridge.py` stands in for the hardware. It publishes `SensorData` and
  `ImuData` on the same ports and topics as Core's telemetry and IMU nodes,
  using Core's own Python `Publisher`/`Subscriber` and generated protos. It
  takes `ThrusterCommand` and does the thruster mapping onto the 8 thrusters,
  since Core has no thruster mapping yet.
* X19-Surface's `depth_subscriber_node` already listens on telemetry :5555,
  so the surface UI shows the simulated depth unchanged. The
  `depthControlCommand` the autonomy sends is the same message the surface sends.
* The Core-side changes (finished `depth_control` node, `thruster_control`
  node, `straight_line` node, protos, CMake) are listed in that branch's
  commits and README.

"Straight" means a constant heading at constant depth, because the ROV has
no position sensor. A cross current still pushes the track sideways, as it
would in the pool.

### Results (`python3 straight_line_test.py`)

Real time, through the full node chain, with noisy sensors and T200 thrust asymmetry:

| scenario | depth err max | heading err max | cross-track max | distance |
|---|---|---|---|---|
| calm | 0.7 cm | 0.43° | 0.5 cm | 7.6 m |
| 0.2 m/s head current | 0.8 cm | 0.47° | 0.3 cm | 6.0 m |
| 0.1 m/s cross current | 0.4 cm | 0.51° | 80 cm (crab, expected) | 7.6 m |
| start rolled 15°, turn across ±180° | 0.6 cm | 0.50° | 1.4 cm | 7.6 m |

Plots are in `results/`, and each node's output is in `results/logs/`.

### What the sim found in Core

* `Controller`: the P term ignored the error (`k_p` alone), there was a
  derivative kick on the first sample, and the filter started at 0 m.
  Unmodified, the ROV drives into the seabed and flips.
* `depth_control.cpp`: `controller->target` was never set. It also didn't
  build (missing `DepthControlSubscriber`, CMake globbed every `main()` into
  one library). Integral windup on dives held the ROV ~4 cm deep.
  Running the PID per loop tick rather than per depth sample sawtoothed the
  heave command.
* Heading at kp 1.0 settles into a ±1.2° limit cycle once real pub/sub
  latency is in the loop. Lowered to 0.3, with an integral zone for turns.

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
  `read_telemetry()` returns them, published by `sim_bridge.py` and sent back to `controller.py` with
  each frame.
* **Thrust mode**: `command_thrust()` drives the thrusters through the
  physics (used by `sim_bridge.py`). The original TCP velocity mode is unchanged.
* **Misc**: the sim retries connecting to the controller, so `run_tcp.sh` has no
  start-order race. The renderer is created lazily, so headless runs need no
  GL. It falls back from EGL to GLFW when EGL is unavailable.
