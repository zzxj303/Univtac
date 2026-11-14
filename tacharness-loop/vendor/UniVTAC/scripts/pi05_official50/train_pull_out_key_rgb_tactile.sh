#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/launch.sh" pull_out_key rgb_tactile "$@"
