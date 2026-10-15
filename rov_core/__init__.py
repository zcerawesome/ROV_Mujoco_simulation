"""Python access to purduerov/X19-Core control code, for use in the sim.

controller.cpp/.hpp are vendored from X19-Core's depth_control branch and
compiled into a small shared library on first import, so the sim exercises
the real C++ PID rather than a Python re-implementation.
"""
from .core_controller import CoreController, build_library
from .thruster_control import ThrusterControl

__all__ = ["CoreController", "ThrusterControl", "build_library"]
