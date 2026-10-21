"""The simulated ROV, on X19-Core's ZeroMQ bus.

Stands in for the vehicle hardware: it publishes what the real sensor nodes
publish, on the same ports and topics, and takes the thruster command the
thruster mapping would. Everything else (depth_control, thruster_control, the
autonomy behaviors) is X19-Core's own code, built from the `core` submodule.

    publishes   SensorData       telemetry   :5555  (depth, accel, gyro)
                ImuData          imu         :5557  (fused quaternion, accel, gyro)
    subscribes  ThrusterCommand  thruster_command :5563  (body frame, [-1, 1] per axis)

X19-Surface's depth_subscriber_node already listens on telemetry :5555, so
the surface UI shows the simulated depth with no changes.

    python3 sim_bridge.py              # viewer window, runs until closed
    python3 sim_bridge.py --headless   # no window (e.g. over SSH)
"""
import argparse
import os
import sys
import time

import numpy as np
import zmq

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "core")
sys.path.insert(0, CORE)   # X19-Core's python packages: src.python.messaging, src.protocols

from simulation import MuJoCoBase                      # noqa: E402  (picks the GL backend)
from src.protocols.python import telemetry_pb2 as pb   # noqa: E402
from src.python.messaging import Publisher, Subscriber  # noqa: E402
from src.python.messaging import endpoints            # noqa: E402

XML_PATH = os.path.join(HERE, "models", "rov.xml")


def vec3(v):
    return pb.Vector3D(x=v[0], y=v[1], z=v[2])


class SimBridge:
    """Connects a MuJoCoBase to the Core bus. Call on_physics_step() once per
    physics step (it is installed as the sim's per-step controller hook)."""

    IMU_HZ = 100.0          # imu_node's default report rate
    TELEMETRY_HZ = 50.0
    COMMAND_TIMEOUT = 0.5   # s without a ThrusterCommand -> thrusters off
    WATER_TEMP_C = 18.0

    def __init__(self, sim, listen_status=False):
        self.sim = sim
        self.telemetry_pub = Publisher(endpoints.TELEMETRY.bind(), endpoints.TELEMETRY.topic)
        self.imu_pub = Publisher(endpoints.IMU.bind(), endpoints.IMU.topic)
        self.thruster_sub = Subscriber(endpoints.THRUSTER_COMMAND.connect(),
                                       endpoints.THRUSTER_COMMAND.topic,
                                       pb.ThrusterCommand, self._on_thruster_command)
        # autonomy status is only for logging/scoring; the sim doesn't act on it
        self.status = None
        self.status_sub = None
        if listen_status:
            self.status_sub = Subscriber(endpoints.AUTONOMY_STATUS.connect(),
                                         endpoints.AUTONOMY_STATUS.topic,
                                         pb.AutonomyStatus, self._on_status)

        self.command = np.zeros(6)
        self._command_wall = -np.inf
        self._next_imu = 0.0
        self._next_telemetry = 0.0
        sim.command_thrust(self.command)
        sim.controller = self.on_physics_step    # simulate() calls this every step

    def _on_thruster_command(self, msg):
        self.command = np.array([msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw])
        self._command_wall = time.monotonic()

    def _on_status(self, msg):
        self.status = msg

    def on_physics_step(self):
        while self.thruster_sub.spin_once(timeout_ms=0):
            pass
        if self.status_sub is not None:
            while self.status_sub.spin_once(timeout_ms=0):
                pass
        if time.monotonic() - self._command_wall > self.COMMAND_TIMEOUT:
            self.command = np.zeros(6)
        self.sim.command_thrust(self.command)

        t = self.sim.data.time
        if t >= self._next_imu or t >= self._next_telemetry:
            tel = self.sim.read_telemetry()
            now_us = int(time.time() * 1e6)
            if t >= self._next_imu:
                self._publish_imu(tel, now_us)
                self._next_imu = t + 1.0 / self.IMU_HZ
            if t >= self._next_telemetry:
                self._publish_telemetry(tel, now_us)
                self._next_telemetry = t + 1.0 / self.TELEMETRY_HZ

    def _publish_imu(self, tel, now_us):
        w, x, y, z = tel["quat"]
        msg = pb.ImuData(timestamp_us=now_us,
                         acceleration=vec3(tel["accel"]),
                         angular_velocity=vec3(tel["gyro"]),
                         orientation=pb.Quaternion(x=x, y=y, z=z, w=w,
                                                   accuracy_rad=self.sim.ATTITUDE_NOISE))
        self.imu_pub.publish(msg)

    def _publish_telemetry(self, tel, now_us):
        msg = pb.SensorData(timestamp_us=now_us,
                            acceleration=vec3(tel["accel"]),
                            angular_velocity=vec3(tel["gyro"]),
                            depth=tel["depth"], temperature=self.WATER_TEMP_C)
        self.telemetry_pub.publish(msg)

    def run_realtime(self, duration=None, on_tick=None, tick_dt=1 / 60):
        """Headless: advance the physics in step with the wall clock (the Core
        nodes run on wall-clock time). on_tick(sim) is called every tick_dt of
        sim time; returning True stops the loop."""
        start = time.monotonic()
        next_tick = 0.0
        while duration is None or self.sim.data.time < duration:
            target = time.monotonic() - start
            while self.sim.data.time < target:
                self.on_physics_step()
                self.sim.step()
                if self.sim.data.time >= next_tick:
                    next_tick += tick_dt
                    if on_tick is not None and on_tick(self.sim):
                        return
            time.sleep(0.001)

    def close(self):
        for s in (self.telemetry_pub, self.imu_pub, self.thruster_sub, self.status_sub):
            if s is not None:
                s.socket.setsockopt(zmq.LINGER, 0)
                s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no viewer window")
    ap.add_argument("--current", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("VX", "VY", "VZ"), help="water current, world frame, m/s")
    args = ap.parse_args()

    sim = MuJoCoBase(XML_PATH, current=args.current)
    bridge = SimBridge(sim)
    print(f"sim_bridge: publishing {endpoints.TELEMETRY.topic} :{endpoints.TELEMETRY.port}, "
          f"{endpoints.IMU.topic} :{endpoints.IMU.port}; "
          f"listening for {endpoints.THRUSTER_COMMAND.topic} :{endpoints.THRUSTER_COMMAND.port}", flush=True)
    try:
        if args.headless:
            bridge.run_realtime()
        else:
            sim.simulate(realtime=True)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        sim.close()


if __name__ == "__main__":
    main()
