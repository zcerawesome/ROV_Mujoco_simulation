import os
# EGL = offscreen GPU rendering. GLFW cannot create a window on this machine
# (Wayland: "EGL: Failed to get EGL display", X11: "No GLXFBConfigs returned"),
# so frames are rendered offscreen and shown in a Tk window instead.
os.environ.setdefault("MUJOCO_GL", "egl")

import base64
import io
import json
import socket
import struct
import time
import tkinter as tk

import numpy as np
import mujoco as mj
import mediapy as media
from PIL import Image, ImageTk
import cv_processor as cvp
from movement_sim import Movement

# The anaconda ffmpeg on PATH (4.3, openh264-only) fails to encode h264;
# point mediapy at the system ffmpeg (6.1, libx264) instead.
if os.path.exists("/usr/bin/ffmpeg"):
    media.set_ffmpeg("/usr/bin/ffmpeg")

ROV_CAMERA = ["front", 'down']

class MuJoCoBase():
    # restoring spring/damper for _jiggle's roll & pitch, in N*m per rad and per rad/s.
    # Without these, nothing in this zero-gravity model pulls tilt back toward level,
    # so the sinusoidal roll/pitch drive alone accumulates into a slow, unbounded drift.
    JIGGLE_K = 0.2
    JIGGLE_D = 0.1

    def __init__(self, xml_path, width=960, height=540, controller_host=None, controller_port=65432):
        self.model = mj.MjModel.from_xml_path(xml_path)   # MuJoCo model
        self.data = mj.MjData(self.model)                 # MuJoCo data
        self.renderer = mj.Renderer(self.model, height=height, width=width)
        self.width, self.height = width, height

        self.cam = mj.MjvCamera()                         # Abstract camera
        self.opt = mj.MjvOption()                         # visualization options
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)

        mj.mj_resetData(self.model, self.data)

        # Aim camera
        self.cam.azimuth = 135
        self.cam.distance = 1.1
        self.cam.elevation = -30
        self.cam.lookat = [0, 0, 0]
        self.active_cam = self.cam

        self.paused = False
        self._drag = None
        self._running = False
        self.movement = Movement()
        self.bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base_link")

        self._build_allocation()

        self._sock = None
        if controller_host is not None:
            self.connect_to_controller(controller_host, controller_port)

    # ---------- controller link ----------
    def connect_to_controller(self, host="localhost", port=65432):
        """Connect to controller.py's server, for run_networked()'s lockstep
        movement-in / frame-out exchange."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))
        self._sock_file = self._sock.makefile("r")
        print(f"Connected to controller at {host}:{port}")

    def run_networked(self, physics_dt=1 / 60, max_steps=None):
        """Lockstep loop: block for one movement vector from the controller,
        apply it, advance the physics by physics_dt, send the rendered frame
        back, and also show it in a local tkinter window (orbit/pan/zoom,
        q/Esc to quit). Requires connect_to_controller() first."""
        assert self._sock is not None, "connect_to_controller() must be called first"

        root = tk.Tk()
        root.title("ROV - MuJoCo (networked)")
        root.resizable(False, False)
        label = tk.Label(root, borderwidth=0)
        label.pack()
        status = tk.Label(root, anchor="w", font=("monospace", 9))
        status.pack(fill="x")

        label.bind("<ButtonPress-1>", lambda e: self._press(e, "orbit"))
        label.bind("<ButtonPress-3>", lambda e: self._press(e, "pan"))
        label.bind("<B1-Motion>", self._motion)
        label.bind("<B3-Motion>", self._motion)
        label.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        label.bind("<ButtonRelease-3>", lambda e: setattr(self, "_drag", None))
        root.bind("<Button-4>", lambda e: self._zoom(-1))
        root.bind("<Button-5>", lambda e: self._zoom(+1))
        root.bind("<MouseWheel>", lambda e: self._zoom(-1 if e.delta > 0 else +1))
        root.bind("q", lambda e: setattr(self, "_running", False))
        root.bind("<Escape>", lambda e: setattr(self, "_running", False))
        root.bind("1", lambda e: setattr(self, "active_cam", self.cam))
        root.bind("2", lambda e: setattr(self, "active_cam", "tracking"))
        root.bind("3", lambda e: setattr(self, "active_cam", "front"))
        root.bind("4", lambda e: setattr(self, "active_cam", "down"))
        self._running = True
        root.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "_running", False))

        steps = 0
        try:
            while self._running and (max_steps is None or steps < max_steps):
                line = self._sock_file.readline()
                if not line:
                    break   # controller closed the connection_running
                line = line.strip()
                if not line:
                    continue
                try:
                    x, y, z, rx, ry, rz = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                self.movement.move(x, y, z, rx, ry, rz)
                self.data.qvel[0:6] = self.movement.get_move()

                target = self.data.time + physics_dt
                while self.data.time < target:
                    mj.mj_step(self.model, self.data)

                frame = self._frame()
                send_data = {cam: self._frame(cam) for cam in ROV_CAMERA}
                self._send_frame(send_data)

                photo = ImageTk.PhotoImage(Image.fromarray(frame))
                label.configure(image=photo)
                label.image = photo                        # keep a reference alive
                status.configure(text=self._status())
                root.update_idletasks()
                root.update()

                steps += 1
        except tk.TclError:
            pass                                            # window closed mid-frame
        finally:
            self._running = False
            try:
                root.destroy()
            except tk.TclError:
                pass

    def _send_frame(self, frames):
        """Send a dict of rendered frames (e.g. {"front": front, "down": down})
        back as one length-prefixed JSON payload of base64-encoded JPEGs."""
        encoded = {}
        for name, frame in frames.items():
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG", quality=80)
            encoded[name] = base64.b64encode(buf.getvalue()).decode("ascii")
        data = json.dumps(encoded).encode("utf-8")
        self._sock.sendall(struct.pack(">I", len(data)) + data)

    # ---------- thruster allocation ----------
    def _build_allocation(self):
        """6xN wrench matrix from the thruster sites, and its pseudo-inverse."""
        mj.mj_forward(self.model, self.data)
        cols = []
        for i in range(self.model.nu):
            sid = self.model.actuator_trnid[i, 0]
            axis = self.data.site_xmat[sid].reshape(3, 3)[:, 2]   # thrust along site +z
            arm = self.data.site_xpos[sid] - self.data.xipos[self.bid]   # lever arm about CoM
            cols.append(np.concatenate([axis, np.cross(arm, axis)]))
        self.W = np.array(cols).T                                 # 6 x nu
        self.W_pinv = np.linalg.pinv(self.W)

    def set_wrench(self, fx=0.0, fy=0.0, fz=0.0, mx=0.0, my=0.0, mz=0.0):
        """Command a body-frame wrench; least-squares allocated across the thrusters."""
        ctrl = self.W_pinv @ np.array([fx, fy, fz, mx, my, mz], dtype=float)
        self.data.ctrl[:] = np.clip(ctrl, -1.0, 1.0)

    def controller(self):
        """Per-step hook used by simulate() (the standalone tkinter viewer).
        Not used by run_networked(), which drives mj_step itself."""
        self.data.qvel[0:6] = self.movement.get_move()

    def _jiggle(self):
        """Gentle sinusoidal push so the ROV drifts and rocks in the water.
        Two mismatched frequencies keep it from looking like a metronome.
        Roll/pitch also get a spring-damper back toward level: this model has
        zero gravity/buoyancy so nothing else restores orientation, and even
        driving roll alone couples into pitch/yaw through the rigid-body
        inertia tensor (three distinct principal moments), so both axes need
        the restoring term or the vehicle slowly tumbles."""
        t = self.data.time
        roll_drive = 0.015 * np.sin(0.9 * t + 0.4)

        up = self.data.xmat[self.bid].reshape(3, 3)[:, 2]   # body +z axis, in world frame
        w = self.data.qvel[3:6]                              # angular velocity, world frame
        self.data.xfrc_applied[self.bid, 3] = roll_drive + self.JIGGLE_K * up[1] - self.JIGGLE_D * w[0]
        self.data.xfrc_applied[self.bid, 4] =             - self.JIGGLE_K * up[0] - self.JIGGLE_D * w[1]

    # ---------- offscreen frame ----------
    def _frame(self, camera=None):
        cam = self.active_cam if camera is None else camera
        self.renderer.update_scene(self.data, camera=cam, scene_option=self.opt)
        return self.renderer.render()

    def render(self, out_path="render.png"):
        """Save a single still frame."""
        mj.mj_forward(self.model, self.data)
        Image.fromarray(self._frame()).save(out_path)
        print(f"Saved render to {out_path}")

    # ---------- live window ----------
    def simulate(self, move=None, realtime=True, max_seconds=None, video_path=None):
        """Run the interactive window. If video_path is given, every displayed
        frame is also recorded and saved as a video when the window closes."""
        frames = [] if video_path is not None else None

        root = tk.Tk()
        root.title("ROV - MuJoCo")
        root.resizable(False, False)
        label = tk.Label(root, borderwidth=0)
        label.pack()
        status = tk.Label(root, anchor="w", font=("monospace", 9))
        status.pack(fill="x")

        # mouse: drag orbits, right-drag pans, wheel zooms
        label.bind("<ButtonPress-1>", lambda e: self._press(e, "orbit"))
        label.bind("<ButtonPress-3>", lambda e: self._press(e, "pan"))
        label.bind("<B1-Motion>", self._motion)
        label.bind("<B3-Motion>", self._motion)
        label.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        label.bind("<ButtonRelease-3>", lambda e: setattr(self, "_drag", None))
        root.bind("<Button-4>", lambda e: self._zoom(-1))
        root.bind("<Button-5>", lambda e: self._zoom(+1))
        root.bind("<MouseWheel>", lambda e: self._zoom(-1 if e.delta > 0 else +1))

        # keys: space pauses, r resets, q/Esc quits
        root.bind("<space>", lambda e: setattr(self, "paused", not self.paused))
        root.bind("r", lambda e: mj.mj_resetData(self.model, self.data))
        root.bind("q", lambda e: setattr(self, "_running", False))
        root.bind("<Escape>", lambda e: setattr(self, "_running", False))
        root.bind("1", lambda e: setattr(self, "active_cam", self.cam))      # free orbit
        root.bind("2", lambda e: setattr(self, "active_cam", "tracking"))
        root.bind("3", lambda e: setattr(self, "active_cam", "front"))
        root.bind("4", lambda e: setattr(self, "active_cam", "down"))
        self._running = True
        root.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "_running", False))
        

        wall0 = time.time()
        try:
            while self._running:
                if not self.paused:
                    # step until sim time catches up with wall-clock time
                    target = (time.time() - wall0) if realtime else self.data.time + 1 / 60
                    steps = 0
                    while self.data.time < target and steps < 200:
                        # self._jiggle()
                        self.controller()
                        mj.mj_step(self.model, self.data)
                        steps += 1

                frame = self._frame()
                # cvp.process_pipe(frame)
                if frames is not None:
                    frames.append(frame)
                photo = ImageTk.PhotoImage(Image.fromarray(frame))
                label.configure(image=photo)
                label.image = photo                       # keep a reference alive
                status.configure(text=self._status())

                root.update_idletasks()
                root.update()

                if max_seconds is not None and self.data.time >= max_seconds:
                    break
                time.sleep(1 / 120)
        except tk.TclError:
            pass                                          # window closed mid-frame
        finally:
            self._running = False
            try:
                root.destroy()
            except tk.TclError:
                pass

        if frames:
            fps = len(frames) / (time.time() - wall0)
            media.write_video(video_path, frames, fps=fps)
            print(f"Saved video to {video_path}")

    def _status(self):
        p = self.data.qpos[:3]
        return ("t=%6.2fs   pos=(% .2f % .2f % .2f)   %s"
                % (self.data.time, p[0], p[1], p[2],
                   "PAUSED" if self.paused else "running"))

    def _press(self, event, mode):
        self._drag = (mode, event.x, event.y)

    def _motion(self, event):
        if self._drag is None:
            return
        mode, lastx, lasty = self._drag
        dx, dy = event.x - lastx, event.y - lasty
        if mode == "orbit":
            self.cam.azimuth = (self.cam.azimuth - 0.4 * dx) % 360
            self.cam.elevation = float(np.clip(self.cam.elevation - 0.4 * dy, -89, 89))
        else:
            a = np.deg2rad(self.cam.azimuth)
            scale = 0.002 * self.cam.distance
            right = np.array([-np.sin(a), np.cos(a), 0.0])
            self.cam.lookat[:] = (np.asarray(self.cam.lookat)
                                  + right * (-dx * scale)
                                  + np.array([0.0, 0.0, dy * scale]))
        self._drag = (mode, event.x, event.y)

    def _zoom(self, direction):
        self.cam.distance = float(np.clip(self.cam.distance * (1.1 ** direction), 0.2, 10.0))

    def close(self):
        self.renderer.close()
        if self._sock is not None:
            self._sock.close()
