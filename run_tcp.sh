#!/usr/bin/env bash
# Original TCP pipeline: controller.py (OAK-D IMU-driven spin) + Main.py viewer.
cd "$(dirname "$0")"
python3 controller.py & python3 Main.py
