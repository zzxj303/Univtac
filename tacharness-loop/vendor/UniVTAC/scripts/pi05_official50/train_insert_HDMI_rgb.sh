#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/launch.sh" insert_HDMI rgb "$@"
