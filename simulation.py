import os
import subprocess
import sys


def _pick_gl_backend():
    """EGL = offscreen GPU rendering. GLFW cannot create a window on some
    machines (Wayland: "EGL: Failed to get EGL display", X11: "No GLXFBConfigs
    returned"), so frames are rendered offscreen and shown in a Tk window
    instead. Machines without a working EGL driver (no GPU, some VMs) can
    usually still render through GLFW. MuJoCo reads MUJOCO_GL at import time,
    so probe EGL in a throwaway process and fall back to GLFW. An explicit
    MUJOCO_GL always wins."""
    if "MUJOCO_GL" in os.environ:
        return
    probe = ("import os; os.environ['MUJOCO_GL'] = 'egl'; "
             "from mujoco.egl import GLContext; GLContext(16, 16).make_current()")
    try:
        ok = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    os.environ["MUJOCO_GL"] = "egl" if ok else "glfw"


_pick_gl_backend()

import base64
import io
import json
import socket
import struct
import time

import numpy as np
import mujoco as mj
import mediapy as media
from PIL import Image
from movement_sim import Movement


def quat_to_rpy(q):
    """(roll, pitch, yaw) in radians, ZYX convention, from (w, x, y, z)."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])

# The anaconda ffmpeg on PATH (4.3, openh264-only) fails to encode h264;
# point mediapy at the system ffmpeg (6.1, libx264) instead.
# With no system ffmpeg at all, fall back to the static build from imageio-ffmpeg.
if os.path.exists("/usr/bin/ffmpeg"):
    media.set_ffmpeg("/usr/bin/ffmpeg")
else:
    try:
        import imageio_ffmpeg
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass

ROV_CAMERA = ["front", 'down']

class MuJoCoBase():
    # sensor noise (1 sigma): a Bar30-class pressure sensor resolves ~2 mm of
    # water, and a BNO08x-class IMU's fused attitude jitters by ~0.2 deg
    DEPTH_NOISE = 0.002        # m
    ATTITUDE_NOISE = 0.0035    # rad
    GYRO_NOISE = 0.005         # rad/s
    ACCEL_NOISE = 0.05         # m/s^2

    def __init__(self, xml_path, width=960, height=540, controller_host=None, controller_port=65432,
                 current=(0.0, 0.0, 0.0), sensor_noise=True, seed=0):
        self.model = mj.MjModel.from_xml_path(xml_path)   # MuJoCo model
        self.data = mj.MjData(self.model)                 # MuJoCo data
        self._renderer = None                             # created on first render
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

        self._load_environment(current)
        self.sensor_noise = sensor_noise
        self._rng = np.random.default_rng(seed)
        self.drive_mode = "velocity"                      # or "thrust"; see step()
        self.thrust_cmd = np.zeros(self.model.nu)         # requested, before motor lag
        self.jiggle = False                               # gentle rocking, see _jiggle()

        self._build_allocation()

        self._sock = None
        if controller_host is not None:
            self.connect_to_controller(controller_host, controller_port)

    # ---------- controller link ----------
    def connect_to_controller(self, host="localhost", port=65432, timeout=30.0):
        """Connect to controller.py's server, for run_networked()'s lockstep
        movement-in / frame-out exchange. Retries until `timeout` so the two
        processes can be started in either order (run.sh starts both at once)."""
        deadline = time.time() + timeout
        while True:
            try:
                self._sock = socket.create_connection((host, port))
                break
            except ConnectionRefusedError:
                if time.time() > deadline:
                    raise
                time.sleep(0.25)
        self._sock_file = self._sock.makefile("r")
        print(f"Connected to controller at {host}:{port}")

    def run_networked(self, physics_dt=1 / 60, max_steps=None):
        """Lockstep loop: block for one movement vector from the controller,
        apply it, advance the physics by physics_dt, send the rendered frame
        back, and also show it in a local tkinter window (orbit/pan/zoom,
        q/Esc to quit). Requires connect_to_controller() first."""
        assert self._sock is not None, "connect_to_controller() must be called first"
        import tkinter as tk
        from PIL import ImageTk

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
                # A bare [x, y, z, rx, ry, rz] list is a velocity command (the
                # original protocol). {"thrust": [x, y, z, roll, pitch, yaw]} is
                # a normalized body-frame command driven through the thrusters.
                try:
                    msg = json.loads(line)
                    if isinstance(msg, dict):
                        self.command_thrust(msg["thrust"])
                    else:
                        self.command_velocity(*msg)
                except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                    continue

                target = self.data.time + physics_dt
                while self.data.time < target:
                    self.step()

                frame = self._frame()
                send_data = {cam: self._frame(cam) for cam in ROV_CAMERA}
                self._send_frame(send_data, self.read_telemetry())

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

    def _send_frame(self, frames, telemetry=None):
        """Send a dict of rendered frames (e.g. {"front": front, "down": down})
        back as one length-prefixed JSON payload of base64-encoded JPEGs. The
        sensor readings ride along under the reserved "telemetry" key."""
        encoded = {}
        for name, frame in frames.items():
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG", quality=80)
            encoded[name] = base64.b64encode(buf.getvalue()).decode("ascii")
        if telemetry is not None:
            encoded["telemetry"] = telemetry
        data = json.dumps(encoded).encode("utf-8")
        self._sock.sendall(struct.pack(">I", len(data)) + data)

    # ---------- thruster allocation ----------
    def _build_allocation(self):
        """6xN wrench matrix from the thruster sites, and its pseudo-inverse.
        Columns are in N and N*m per unit ctrl, i.e. scaled by the actuator
        gear (thrust at ctrl = 1), so set_wrench() really takes SI units.
        Built in the body frame, from the model's reset pose."""
        mj.mj_forward(self.model, self.data)
        R = self.data.xmat[self.bid].reshape(3, 3)
        cols = []
        for i in range(self.model.nu):
            sid = self.model.actuator_trnid[i, 0]
            gear = self.model.actuator_gear[i, 2]
            axis = R.T @ self.data.site_xmat[sid].reshape(3, 3)[:, 2]   # thrust along site +z
            arm = R.T @ (self.data.site_xpos[sid] - self.data.xipos[self.bid])   # lever arm about CoM
            cols.append(gear * np.concatenate([axis, np.cross(arm, axis)]))
        self.W = np.array(cols).T                                 # 6 x nu
        self.W_pinv = np.linalg.pinv(self.W)

        # Full authority of each axis on its own (N, N*m): the largest wrench
        # along that axis before some thruster saturates. Lets callers send
        # normalized [-1, 1] commands that mean the same thing on every axis.
        self.axis_max = np.array([1.0 / np.abs(self.W_pinv[:, k]).max() for k in range(6)])

    def set_wrench(self, fx=0.0, fy=0.0, fz=0.0, mx=0.0, my=0.0, mz=0.0):
        """Command a body-frame wrench; least-squares allocated across the thrusters.
        Thrusters asked to push backwards get a proportionally larger command,
        since they are weaker in reverse (thruster_reverse_ratio) -- without
        this, e.g. a pure surge leaks into yaw and sway on this pinwheel layout.
        If any thruster would saturate, the whole vector is scaled down rather
        than clipping thrusters one by one, which would change the direction
        of the wrench (e.g. a dive command turning into a dive-and-yaw)."""
        ctrl = self.W_pinv @ np.array([fx, fy, fz, mx, my, mz], dtype=float)
        ctrl = np.where(ctrl < 0, ctrl / self.reverse_ratio, ctrl)
        peak = np.abs(ctrl).max()
        if peak > 1.0:
            ctrl /= peak
        self.thrust_cmd[:] = ctrl

    def command_thrust(self, cmd):
        """Normalized body-frame command [x, y, z, roll, pitch, yaw] in [-1, 1],
        1.0 being that axis's full authority (self.axis_max), driven through
        the thrusters. Switches the sim into thrust mode."""
        self.drive_mode = "thrust"
        self.set_wrench(*(np.clip(np.asarray(cmd, dtype=float), -1.0, 1.0) * self.axis_max))

    def command_velocity(self, x, y, z, rx, ry, rz):
        """Original kinematic mode: the body velocity is set directly each step
        (world linear, body angular) and the thrusters are idle."""
        self.drive_mode = "velocity"
        self.thrust_cmd[:] = 0.0
        self.movement.move(x, y, z, rx, ry, rz)

    def controller(self):
        """Per-step hook used by simulate() (the standalone tkinter viewer)."""
        if self.drive_mode == "velocity":
            self.data.qvel[0:6] = self.movement.get_move()

    # ---------- physics ----------
    def _numeric(self, name, default):
        nid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_NUMERIC, name)
        if nid < 0:
            return np.atleast_1d(np.asarray(default, dtype=float))
        adr, size = self.model.numeric_adr[nid], self.model.numeric_size[nid]
        return self.model.numeric_data[adr:adr + size].copy()

    def _load_environment(self, current):
        self.water_density = float(self._numeric("water_density", 1000.0)[0])
        self.damping_linear = self._numeric("damping_linear", np.zeros(6))
        self.damping_quadratic = self._numeric("damping_quadratic", np.zeros(6))
        self.buoyancy_volume = float(self._numeric("buoyancy_volume", 0.0)[0])
        self.cob_offset = self._numeric("cob_offset", [0, 0, 0])
        self.hull_height = float(self._numeric("hull_height", 0.17)[0])
        self.reverse_ratio = float(self._numeric("thruster_reverse_ratio", 1.0)[0])
        self.thruster_tau = float(self._numeric("thruster_time_constant", 0.0)[0])
        self.set_current(current)

    def set_current(self, current):
        """Water current (world frame, m/s). It acts through the drag model
        (the hull is dragged by its velocity relative to the water), so it
        pushes the vehicle the way a real current does rather than a fixed force."""
        self.current = np.asarray(current, dtype=float)

    def _apply_hydrodynamics(self):
        """Linear + quadratic damping on the velocity relative to the water,
        in the body frame (the hull's drag differs per axis), applied at the
        CoM. Adds to the buoyancy wrench already in xfrc_applied."""
        R = self.data.xmat[self.bid].reshape(3, 3)
        v = R.T @ (self.data.qvel[0:3] - self.current)    # freejoint: world linear vel
        w = self.data.qvel[3:6]                            # freejoint: body angular vel
        nu = np.concatenate([v, w])
        tau = -(self.damping_linear * nu + self.damping_quadratic * np.abs(nu) * nu)
        self.data.xfrc_applied[self.bid, :3] += R @ tau[:3]
        self.data.xfrc_applied[self.bid, 3:] += R @ tau[3:]

    def _apply_buoyancy(self):
        """Archimedes' force at the centre of buoyancy. MuJoCo has no
        buoyancy of its own. It fades out linearly as the hull breaches the
        surface (z = 0), so a positively buoyant ROV floats at the top instead
        of rising forever. CoB above CoM -> a righting moment in roll/pitch."""
        g = -self.model.opt.gravity[2]
        z = self.data.xipos[self.bid, 2]
        submerged = np.clip(0.5 - z / self.hull_height, 0.0, 1.0)
        force = np.array([0.0, 0.0, self.water_density * self.buoyancy_volume * g * submerged])
        arm = self.data.xmat[self.bid].reshape(3, 3) @ self.cob_offset
        self.data.xfrc_applied[self.bid, :3] = force
        self.data.xfrc_applied[self.bid, 3:] = np.cross(arm, force)

    def _update_thrusters(self):
        """Motor spin-up lag, then T200-style asymmetry: reverse thrust is
        weaker, so a symmetric allocation leaves a small residual that the
        closed loop has to absorb, as on the real vehicle."""
        dt = self.model.opt.timestep
        alpha = 1.0 if self.thruster_tau <= 0 else dt / (self.thruster_tau + dt)
        ctrl = self.data.ctrl
        ctrl += alpha * (self.thrust_cmd - ctrl)
        self._thrust_applied = np.where(ctrl < 0, ctrl * self.reverse_ratio, ctrl)

    def step(self):
        """One physics step with the water model applied. Use this instead of
        a bare mj.mj_step so buoyancy and the thrusters are accounted for."""
        self._apply_buoyancy()
        self._apply_hydrodynamics()
        if self.jiggle:
            self._jiggle()
        self._update_thrusters()
        if self.drive_mode == "velocity":
            self.data.qvel[0:6] = self.movement.get_move()
        # data.ctrl holds the lagged command; swap in the asymmetric force for
        # the step only, so the motor lag state isn't polluted by the scaling
        lagged = self.data.ctrl.copy()
        self.data.ctrl[:] = self._thrust_applied
        mj.mj_step(self.model, self.data)
        self.data.ctrl[:] = lagged

    def _jiggle(self):
        """Gentle sinusoidal push so the ROV drifts and rocks in the water.
        Adds to the buoyancy torque; the CoB above the CoM supplies the
        restoring moment that keeps the rocking bounded."""
        t = self.data.time
        self.data.xfrc_applied[self.bid, 3] += 0.015 * np.sin(0.9 * t + 0.4)
        self.data.xfrc_applied[self.bid, 4] += 0.010 * np.sin(1.3 * t)

    # ---------- sensors ----------
    def read_telemetry(self):
        """What the ROV itself can measure, with sensor noise, plus the
        ground truth for logging and scoring (never feed "truth" to a
        controller -- the real vehicle has no such thing).

        depth: m below the surface, from the pressure sensor
        quat: (w, x, y, z) attitude, body -> world; rpy: the same in rad
        gyro: body angular rate, rad/s; accel: specific force, m/s^2"""
        sd = self.data.sensordata
        def sensor(name):
            sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
            adr, dim = self.model.sensor_adr[sid], self.model.sensor_dim[sid]
            return sd[adr:adr + dim].copy()

        quat = sensor("imu_quat")
        gyro, accel = sensor("imu_gyro"), sensor("imu_accel")
        depth = -sensor("pressure_pos")[2]
        if self.sensor_noise:
            n = self._rng.normal
            depth += n(0, self.DEPTH_NOISE)
            dq = np.zeros(4)
            mj.mju_axisAngle2Quat(dq, n(0, 1, 3), np.abs(n(0, self.ATTITUDE_NOISE)))
            mj.mju_mulQuat(quat, quat.copy(), dq)
            gyro += n(0, self.GYRO_NOISE, 3)
            accel += n(0, self.ACCEL_NOISE, 3)
        return {
            "time": float(self.data.time),
            "depth": float(depth),
            "quat": quat.tolist(),
            "rpy": quat_to_rpy(quat).tolist(),
            "gyro": gyro.tolist(),
            "accel": accel.tolist(),
            "axis_max": self.axis_max.tolist(),
            "truth": {
                "pos": self.data.qpos[0:3].tolist(),
                "quat": self.data.qpos[3:7].tolist(),
                "vel": self.data.qvel[0:3].tolist(),
            },
        }

    # ---------- offscreen frame ----------
    @property
    def renderer(self):
        if self._renderer is None:
            self._renderer = mj.Renderer(self.model, height=self.height, width=self.width)
        return self._renderer

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
        import tkinter as tk
        from PIL import ImageTk
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
                        self.controller()
                        self.step()
                        steps += 1

                frame = self._frame()
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
        yaw = np.degrees(quat_to_rpy(self.data.qpos[3:7])[2])
        return ("t=%6.2fs   pos=(% .2f % .2f % .2f)   depth=%.2fm   heading=% .1fdeg   %s"
                % (self.data.time, p[0], p[1], p[2], -p[2], yaw,
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
        if self._renderer is not None:
            self._renderer.close()
        if self._sock is not None:
            self._sock.close()
