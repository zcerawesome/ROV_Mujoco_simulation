#!/usr/bin/env bash
# The simulated ROV driven by X19-Core's own nodes over ZeroMQ: builds the
# Core nodes from the `core` submodule, then starts them and the sim with
# Core's launcher (see sim_launch.yaml). Ctrl+C stops everything.
#
#   ./run.sh                                     1.5 m, surge 0.5, 8 s, current heading
#   ./run.sh --depth 2 --surge 0.6 --cruise-time 10 --heading-deg 90
#   SIM_ARGS="--headless" ./run.sh               no viewer window
#   SIM_ARGS="--current 0 0.1 0" ./run.sh        cross current
set -e
cd "$(dirname "$0")"
export PYTHON="${PYTHON:-python3}"

[ -f core/CMakeLists.txt ] || git submodule update --init
[ -f core/build/CMakeCache.txt ] || cmake -S core -B core/build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build core/build -j --target depth_control thruster_control straight_line > /dev/null

export STRAIGHT_LINE_ARGS="$*"
exec "$PYTHON" core/scripts/launch.py sim_launch.yaml
