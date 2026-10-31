#!/usr/bin/env bash
set -euo pipefail

python -m pip install --user --break-system-packages -r /work/env/requirements.txt
python -m pip install --user --break-system-packages -e /work
