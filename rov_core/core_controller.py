import ctypes
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCES = [os.path.join(_HERE, f) for f in ("controller.cpp", "controller_capi.cpp")]
_HEADERS = [os.path.join(_HERE, "controller.hpp")]
_LIB_PATH = os.path.join(_HERE, "build", "libcorecontroller.so")

_lib = None


def build_library(force=False):
    """Compile the vendored core controller with g++ if the .so is missing or
    older than its sources. Returns the path to the shared library."""
    newest_src = max(os.path.getmtime(p) for p in _SOURCES + _HEADERS)
    if (not force and os.path.exists(_LIB_PATH)
            and os.path.getmtime(_LIB_PATH) >= newest_src):
        return _LIB_PATH
    os.makedirs(os.path.dirname(_LIB_PATH), exist_ok=True)
    cxx = os.environ.get("CXX", "g++")
    cmd = [cxx, "-std=c++20", "-O2", "-shared", "-fPIC", "-Wall",
           "-o", _LIB_PATH, *_SOURCES]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(f"'{cxx}' not found: a C++ compiler is needed to build "
                           "the X19-Core controller (apt install g++)") from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"building the core controller failed:\n{e.stderr}") from None
    return _LIB_PATH


def _load():
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(build_library())
    p, d = ctypes.c_void_p, ctypes.c_double
    sigs = {
        "ctrl_new": ([d, d, d], p),
        "ctrl_free": ([p], None),
        "ctrl_set_target": ([p, d], None),
        "ctrl_get_target": ([p], d),
        "ctrl_set_weights": ([p, d, d, d], None),
        "ctrl_record_measurement": ([p, d], None),
        "ctrl_get_measurement": ([p], d),
        "ctrl_compute": ([p, d, d], d),
        "ctrl_compute_filtered": ([p, d], d),
        "ctrl_get_integral": ([p], d),
        "ctrl_reset_integral": ([p], None),
    }
    for name, (args, res) in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    _lib = lib
    return lib


class CoreController:
    """Thin wrapper over X19-Core's `Controller` (PID + EMA filter, output
    clamped to [-1, 1]). Used for depth exactly as depth_control.cpp does,
    and reused for the heading/attitude loops of the autonomy behaviors."""

    def __init__(self, k_p, k_i, k_d, target=0.0):
        self._lib = _load()
        self._ptr = self._lib.ctrl_new(k_p, k_i, k_d)
        self.target = target

    @property
    def target(self):
        return self._lib.ctrl_get_target(self._ptr)

    @target.setter
    def target(self, value):
        self._lib.ctrl_set_target(self._ptr, float(value))

    @property
    def integral(self):
        return self._lib.ctrl_get_integral(self._ptr)

    def set_weights(self, k_p, k_i, k_d):
        self._lib.ctrl_set_weights(self._ptr, k_p, k_i, k_d)

    def record_measurement(self, measurement):
        self._lib.ctrl_record_measurement(self._ptr, float(measurement))

    def get_measurement(self):
        return self._lib.ctrl_get_measurement(self._ptr)

    def compute(self, measurement, dt):
        return self._lib.ctrl_compute(self._ptr, float(measurement), float(dt))

    def compute_filtered(self, dt):
        return self._lib.ctrl_compute_filtered(self._ptr, float(dt))

    def reset_integral(self):
        self._lib.ctrl_reset_integral(self._ptr)

    def __del__(self):
        ptr, self._ptr = getattr(self, "_ptr", None), None
        if ptr:
            self._lib.ctrl_free(ptr)
