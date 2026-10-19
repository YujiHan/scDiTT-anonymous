#!/usr/bin/env bash
# ============================================================================
# scDiTT -- conda interpreter / runtime environment configuration
# ----------------------------------------------------------------------------
# Every exps/*/run/*.sh `source`s this file to obtain ${PYTHON}.
# The benefit: when moving to a new server you only edit the single line below
# and every script picks it up (no need to edit them one by one).
#
# >>>  On a new server, change only this line: point it at the python
# >>>  interpreter of the target conda environment.
#   Find the path with:  conda run -n scDiTT which python
#                        (or conda activate scDiTT && which python)
#   Override temporarily: PYTHON=/path/to/python bash exps/task1_ts_pred/run/train.sh ...
# Anonymised submission version: the original default was the absolute path of
# the authors' local conda interpreter and has been cleared.
# While it is empty the downstream scripts have no interpreter, so fill it in
# here or override it with PYTHON=...
export PYTHON="${PYTHON:-}"
# ============================================================================
