#!/bin/bash

set -e -E -u -o pipefail

source rapids-configure-sccache

rapids-print-env

rapids-generate-version > ./VERSION

sccache --zero-stats

CMAKE_GENERATOR=Ninja \
CONDA_OVERRIDE_CUDA="${RAPIDS_CUDA_VERSION}" \
LEGATERAFT_PACKAGE_VERSION=$(head -1 ./VERSION) \
rapids-conda-retry build \
    --channel legate \
    --channel legate/label/rc \
    --channel legate/label/experimental \
    --channel rapidsai \
    --channel conda-forge \
    --channel nvidia \
    --no-force-upload \
    conda/recipes/legate-raft

sccache --show-adv-stats

# echo package details to logs, to help with debugging
conda search \
    --override-channels \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --info \
        legate-raft
