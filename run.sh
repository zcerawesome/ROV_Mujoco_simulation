#!/usr/bin/env bash
# Controller (TCP server) + simulation (client). The sim retries the
# connection, so start order doesn't matter. Extra args go to controller.py,
# e.g.  ./run.sh --goal straight --depth 2.0   or   ./run.sh --goal spin
cd "$(dirname "$0")"
python3 controller.py "$@" &
CONTROLLER=$!
trap 'kill $CONTROLLER 2>/dev/null' EXIT
python3 Main.py
