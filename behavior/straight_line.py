import math

import numpy as np

from rov_core import CoreController, ThrusterControl


def wrap_angle(a):
    """Wrap to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class StraightLine:
    """Drive the ROV in a straight line, autonomously, on the vehicle's own
    sensors (pressure depth + IMU attitude):

      settle  dive to target_depth and lock the heading, no forward thrust
      cruise  hold depth and heading while surging forward for cruise_time
      stop    surge off, keep holding depth and heading

    Depth runs exactly the loop in X19-Core's depth_control.cpp on the core
    Controller (record_measurement -> target -> compute_filtered). Heading,
    roll and pitch reuse the same Controller class with their own gains, and
    ThrusterControl merges them the way thruster_control.hpp intends.

    There is no position sensor on the vehicle, so "straight" means a
    constant heading at constant depth: a cross current will still push the
    track sideways (crab), exactly as it would the real ROV."""

    # PID weights, (k_p, k_i, k_d) per unit of error (m or rad), output in
    # [-1, 1] of that axis's full authority
    DEPTH_GAINS = (1.2, 0.2, 0.5)
    HEADING_GAINS = (1.0, 0.1, 0.1)
    ROLL_GAINS = (1.2, 0.0, 0.15)
    PITCH_GAINS = (1.2, 0.0, 0.15)

    # Integrate depth error only within this band (m). The core Controller's
    # anti-windup only stops integrating while the output is saturated, and a
    # 1 m dive at ~0.4 output never saturates, so the integral piled up on the
    # way down and held the ROV ~4 cm too deep for tens of seconds afterwards.
    DEPTH_INTEGRAL_ZONE = 0.15

    def __init__(self, axis_max, target_depth=1.5, surge=0.5, cruise_time=8.0,
                 heading=None, settle_tolerance=0.05, settle_hold=2.0, settle_timeout=20.0):
        """heading: rad, or None to lock whatever heading the ROV has when the
        behavior starts. surge: forward command in [-1, 1] during cruise."""
        self.surge = surge
        self.cruise_time = cruise_time
        self.settle_tolerance = settle_tolerance
        self.settle_hold = settle_hold
        self.settle_timeout = settle_timeout

        self.depth = CoreController(*self.DEPTH_GAINS, target=target_depth)
        self.heading = CoreController(*self.HEADING_GAINS)
        self.roll = CoreController(*self.ROLL_GAINS, target=0.0)
        self.pitch = CoreController(*self.PITCH_GAINS, target=0.0)
        self._heading_target = heading
        self.thrusters = ThrusterControl(axis_max)

        self.phase = "settle"
        self._phase_time = 0.0
        self._settled_for = 0.0
        self.last_command = np.zeros(6)

    @property
    def heading_target(self):
        return self._heading_target

    def done(self):
        return self.phase == "stop"

    def _enter(self, phase):
        self.phase = phase
        self._phase_time = 0.0

    def update(self, telemetry, dt):
        """One control tick. telemetry: the sim's read_telemetry() dict (only
        the sensor fields are used, never "truth"). Returns the normalized
        body-frame thrust command [x, y, z, roll, pitch, yaw]."""
        roll, pitch, yaw = telemetry["rpy"]
        if self._heading_target is None:
            self._heading_target = yaw
        self.heading.target = self._heading_target

        # depth: the depth_control.cpp loop
        self.depth.record_measurement(telemetry["depth"])
        if abs(self.depth.target - self.depth.get_measurement()) > self.DEPTH_INTEGRAL_ZONE:
            self.depth.reset_integral()
        depth_cmd = self.depth.compute_filtered(dt)

        # heading: hand the controller a measurement unwrapped around its
        # target, so crossing +-180 deg is a small error, not a 360 deg one
        self.heading.record_measurement(
            self._heading_target - wrap_angle(self._heading_target - yaw))
        yaw_cmd = self.heading.compute_filtered(dt)

        self.roll.record_measurement(roll)
        self.pitch.record_measurement(pitch)
        roll_cmd = self.roll.compute_filtered(dt)
        pitch_cmd = self.pitch.compute_filtered(dt)

        self._phase_time += dt
        depth_err = abs(self.depth.target - self.depth.get_measurement())
        if self.phase == "settle":
            self._settled_for = self._settled_for + dt if depth_err < self.settle_tolerance else 0.0
            if self._settled_for >= self.settle_hold or self._phase_time >= self.settle_timeout:
                self._enter("cruise")
        elif self.phase == "cruise" and self._phase_time >= self.cruise_time:
            self._enter("stop")

        surge = self.surge if self.phase == "cruise" else 0.0
        command = [surge, 0.0, 0.0, roll_cmd, pitch_cmd, yaw_cmd]
        self.last_command = self.thrusters.gen_thruster_command(
            telemetry["quat"], command, depth_cmd)
        return self.last_command
