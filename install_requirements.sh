#!/usr/bin/env bash
# Python deps (the sim uses the official `mujoco` bindings, not mujoco-py),
# plus g++ to build the vendored X19-Core controller and Tk for the viewer.
set -e
command -v g++ >/dev/null || echo "note: install a C++ compiler for rov_core (sudo apt install g++)"
python3 -c "import tkinter" 2>/dev/null || echo "note: install Tk for the viewer (sudo apt install python3-tk)"
pip3 install -r "$(dirname "$0")/requirements.txt"
