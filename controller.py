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
from imu import track
from movement_sim import Movement
from behavior.spin import Spin

class Goals(Enum):
    SPIN=1,
    SPIN_AROUND=2

GOAL = Goals.SPIN

class Controller:
    """TCP server. simulation.py connects to this as a client. Each exchange
    is sequential: send_movement() pushes a [x,y,z,rx,ry,rz] JSON line, then
    recv_frame() blocks for the length-prefixed JSON payload of named,
    base64-encoded JPEG frames (e.g. "front", "down") sent back once the
    simulation has applied that movement."""

    def __init__(self, host="localhost", port=65432) -> None:
        self.movement = Movement()
        self.initial_time = time.time()
        self.time = time.time()
        self.host = host
        self.port = port
        self._server_socket = None
        self._client_socket = None

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


if __name__ == "__main__":
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
