#!/usr/bin/env bash
set -euo pipefail
python download_voc.py --data-dir "${1:-data}"
