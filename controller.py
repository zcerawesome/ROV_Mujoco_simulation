import argparse
import base64
import io
import json
import socket
import struct
import time

import numpy as np
from enum import Enum
from PIL import Image

import cv_processor as cvp
from movement_sim import Movement
from behavior.spin import Spin
from behavior.straight_line import StraightLine

class Goals(Enum):
    SPIN=1,
    SPIN_AROUND=2
    STRAIGHT_LINE=3

class Controller:
    """TCP server. simulation.py connects to this as a client. Each exchange
    is sequential: send_movement() pushes a [x,y,z,rx,ry,rz] JSON line, then
    recv_frame() blocks for the length-prefixed JSON payload of named,
    base64-encoded JPEG frames (e.g. "front", "down") sent back once the
    simulation has applied that movement.

    send_thrust() is the alternative to send_movement(): a normalized
    body-frame thrust command that the sim drives through its thrusters and
    physics, instead of setting the velocity directly. Either way, the sim's
    sensor readings for that step are left in self.telemetry by recv_frame()."""

    def __init__(self, host="localhost", port=65432) -> None:
        self.movement = Movement()
        self.initial_time = time.time()
        self.time = time.time()
        self.host = host
        self.port = port
        self._server_socket = None
        self._client_socket = None
        self.telemetry = None

    def start_server(self):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1)
        print(f"Controller server listening on {self.host}:{self.port}, waiting for simulation...")
        self._client_socket, addr = self._server_socket.accept()
        print(f"Simulation connected from {addr}")

    def send_movement(self, x, y, z, x_rot, y_rot, z_rot):
        move = self.movement.move(x, y, z, x_rot, y_rot, z_rot)
        payload = (json.dumps(move) + "\n").encode("utf-8")
        self._client_socket.sendall(payload)

    def send_thrust(self, cmd):
        """cmd: [x, y, z, roll, pitch, yaw] in [-1, 1], body frame, 1.0 being
        full authority on that axis."""
        payload = (json.dumps({"thrust": [float(c) for c in cmd]}) + "\n").encode("utf-8")
        self._client_socket.sendall(payload)

    def recv_frame(self):
        """Block for one length-prefixed JSON payload of named, base64-encoded
        JPEG frames. Returns a dict of name -> RGB ndarray (e.g. {"front":
        front, "down": down}), or None if the simulation closed the
        connection."""
        header = self._recvn(4)
        if header is None:
            return None
        (length,) = struct.unpack(">I", header)
        data = self._recvn(length)
        if data is None:
            return None
        encoded = json.loads(data.decode("utf-8"))
        self.telemetry = encoded.pop("telemetry", None)
        return {
            name: np.array(Image.open(io.BytesIO(base64.b64decode(b))))
            for name, b in encoded.items()
        }

    def _recvn(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self._client_socket.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def close(self):
        if self._client_socket is not None:
            self._client_socket.close()
        if self._server_socket is not None:
            self._server_socket.close()


def run_straight_line(args):
    """Straight-line autonomy against the sim, using only the sim's sensor
    telemetry (no OAK-D needed): dive to --depth, lock heading, cruise."""
    controller = Controller()
    controller.start_server()
    try:
        controller.send_thrust([0.0] * 6)          # first exchange: get telemetry
        if controller.recv_frame() is None:
            print("simulation disconnected")
            return
        tel = controller.telemetry
        behavior = StraightLine(tel["axis_max"], target_depth=args.depth,
                                surge=args.surge, cruise_time=args.cruise_time)
        last_t = tel["time"]
        while True:
            dt = max(tel["time"] - last_t, 1e-3)
            last_t = tel["time"]
            cmd = behavior.update(tel, dt)
            controller.send_thrust(cmd)
            frames = controller.recv_frame()
            if frames is None:
                print("simulation disconnected")
                break
            tel = controller.telemetry
            print(f"\r[{behavior.phase:6s}] t={tel['time']:6.2f}s depth={tel['depth']:.2f}m "
                  f"heading={np.degrees(tel['rpy'][2]):+6.1f}deg", end="", flush=True)
            if args.show:
                cvp.show("front", frames["front"], bgr=False)
    except KeyboardInterrupt:
        pass
    except (ConnectionResetError, BrokenPipeError):
        print("simulation disconnected")
    finally:
        print()
        controller.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", choices=["straight", "spin"], default="straight",
                    help="straight: straight-line autonomy on sim telemetry; "
                         "spin: the OAK-D IMU-driven spin (needs the camera)")
    ap.add_argument("--depth", type=float, default=1.5, help="straight: target depth (m)")
    ap.add_argument("--surge", type=float, default=0.5, help="straight: forward command [-1, 1]")
    ap.add_argument("--cruise-time", type=float, default=8.0, help="straight: seconds of cruise")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="don't open the OpenCV window with the front camera")
    args = ap.parse_args()
    GOAL = Goals.STRAIGHT_LINE if args.goal == "straight" else Goals.SPIN
    if GOAL == Goals.STRAIGHT_LINE:
        run_straight_line(args)
        raise SystemExit

    from imu import track     # needs depthai + the OAK-D unless replaying a CSV

    # IMU source for track.get_pose(): None = live OAK-D camera, or a path to a
    # Test A CSV log to replay in real time (e.g. "imu/imu_test_a.csv").
    IMU_SOURCE = None

    controller = Controller()
    # First call starts the IMU stream so it calibrates while we wait for the sim.
    track.get_pose(IMU_SOURCE)
    controller.start_server()

    # Fixed forward-drive command, sent automatically every step.
    MOVEMENT = (0, 0, 0, 0, 0, 0.5)

    print("Sending movement automatically (Ctrl+C to quit)")
    video = []
    start_time = time.time()
    spin = Spin()
    try:
        while True:
            controller.send_movement(*MOVEMENT)
            frames = controller.recv_frame()
            if frames is None:
                print("simulation disconnected")
                break
            pose = track.get_pose(IMU_SOURCE)
            if pose.calibrated:
                x, y, z = pose.position
                roll, pitch, yaw = pose.rotation
                # print(f"\rpos=({x:+.2f} {y:+.2f} {z:+.2f}) m  "
                #       f"rot=({roll:+.1f} {pitch:+.1f} {yaw:+.1f}) deg", end="", flush=True)

                if GOAL == Goals.SPIN:
                    spin.current_pos(x, y, z, roll, pitch, yaw)
                    MOVEMENT = spin.current_move()
                    print(MOVEMENT)
                    depthai_camera = track.get_frame()
                    if not spin.done():
                        video.append(cvp.encode_jpeg(depthai_camera))
                    elif not spin.task_complete:
                        elapsed = time.time() - start_time
                        fps = len(video) / elapsed if video and elapsed > 0 else 30.0
                        cvp.save_video(video, 'sim.mp4', fps=fps)
                        spin.complete_task()
            cvp.show("front", frames["front"], bgr=False)
    except KeyboardInterrupt:
        pass
    except (ConnectionResetError, BrokenPipeError):
        print("simulation disconnected")
    finally:
        controller.close()
