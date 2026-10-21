#!/usr/bin/env bash
# Sets up the sim and the X19-Core nodes it runs (the `core` submodule).
set -e
cd "$(dirname "$0")"
git submodule update --init

# Core's C++ nodes: the same system packages as core/scripts/setup.sh, plus Eigen.
# (Eigen is header-only; without sudo, install it anywhere and export
# CMAKE_PREFIX_PATH=<prefix> before building.)
if command -v apt-get >/dev/null; then
  echo "Core build deps: sudo apt-get install -y protobuf-compiler libprotobuf-dev libzmq3-dev cppzmq-dev libeigen3-dev cmake build-essential python3-tk"
fi

pip3 install -r requirements.txt
cmake -S core -B core/build -DCMAKE_BUILD_TYPE=Release
cmake --build core/build -j --target depth_control thruster_control straight_line
