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
from behavior.docking import Docking

class Goals(Enum):
    SPIN=1,
    SPIN_AROUND=2
    NOTHING = 3
    DOCKING = 4

GOAL = Goals.DOCKING

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
    controller = Controller()
    controller.start_server()

    # Fixed forward-drive command, sent automatically every step.
    dock = Docking(0.25)
    MOVEMENT = (0, 0, 0, 0, 0, 0)

    print("Sending movement automatically (Ctrl+C to quit)")
    try:
        while True:
            controller.send_movement(*MOVEMENT)
            frames = controller.recv_frame()
            if frames is None:
                print("simulation disconnected")
                break
            # cvp.show("front", frames["down"], bgr=False)
            if GOAL == Goals.DOCKING:
                front_box = cvp.process_pipe(frames['front'])[0]
                down_box = cvp.process_pipe(frames['down'])[0]
                cvp.show('pipe (front | down)', np.hstack([front_box, down_box]), bgr=False)
                MOVEMENT = dock.process(frames)
                # print(w * h)
                # if w * h < 0.2:
                #     MOVEMENT = (0, 0, -1, 0, 0, 0)
                # else:
                #     MOVEMENT = (0, 0, 0, 0, 0, 0)

    except KeyboardInterrupt:
        pass
    except (ConnectionResetError, BrokenPipeError):
        print("simulation disconnected")
    finally:
        controller.close()
