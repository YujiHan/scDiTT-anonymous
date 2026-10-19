#!/usr/bin/env bash
# ============================================================================
# ----------------------------------------------------------------------------
#
#
#
#   bash run_baseline_umap.sh --methods "PI-SDE scNODE" --datasets "ZB DR"
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/env.sh"

"${PYTHON}" "${SCRIPT_DIR}/baseline_umap_visualize.py" "$@"
