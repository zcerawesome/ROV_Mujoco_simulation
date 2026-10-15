import numpy as np

AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def quat_to_matrix(q):
    """Rotation matrix (body -> world) from a (w, x, y, z) quaternion."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_rpy(q):
    """(roll, pitch, yaw) in radians, ZYX convention, from (w, x, y, z)."""
    w, x, y, z = q / np.linalg.norm(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class ThrusterControl:
    """Python counterpart of X19-Core's thruster_control.hpp stub,
    genThrusterCommand(measuredPose, joystickCommand, depthControlCommand):
    merges a body-frame pilot/autonomy command with the depth controller's
    output and returns one body-frame command for the thruster mapping.

    The depth loop works in the world frame (depth is along gravity), but the
    thrusters push in the body frame, so the depth command is rotated through
    the measured attitude. Without that, any pitch or roll leaks the vertical
    effort into surge/sway.

    All commands are normalized: 1.0 on an axis means that axis's full
    authority, `axis_max` (N for x/y/z, N*m for roll/pitch/yaw)."""

    def __init__(self, axis_max):
        self.axis_max = np.asarray(axis_max, dtype=float)

    def gen_thruster_command(self, orientation_quat, command, depth_command=None):
        """orientation_quat: measured (w, x, y, z), body -> world.
        command: 6 normalized body-frame values [x, y, z, roll, pitch, yaw].
        depth_command: depth controller output in [-1, 1], positive = deeper,
        or None when depth control is disabled (pilot z passes through).
        Returns the 6 normalized body-frame values, each clipped to [-1, 1]."""
        out = np.asarray(command, dtype=float).copy()
        if depth_command is not None:
            R = quat_to_matrix(np.asarray(orientation_quat, dtype=float))
            f_world = np.array([0.0, 0.0, -depth_command * self.axis_max[2]])
            f_body = R.T @ f_world
            out[:3] += f_body / self.axis_max[:3]
        return np.clip(out, -1.0, 1.0)
