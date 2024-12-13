#!/bin/bash

# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# raft empty project template build script

# Abort script on first error
set -e

INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}

PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

BUILD_TYPE=Release
BUILD_DIR="${REPODIR}/cpp/build/"
BUILD_DIRS="${BUILD_DIR} ${REPODIR}/_skbuild ${REPODIR}/build"

RAFT_REPO_REL=${RAFT_REPO_REL:-}
EXTRA_CMAKE_ARGS=${EXTRA_CMAKE_ARGS:-}

if [[ ${RAFT_REPO_REL} != "" ]]; then
  RAFT_REPO_PATH="`readlink -f \"${RAFT_REPO_REL}\"`"
  EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCPM_raft_SOURCE=${RAFT_REPO_PATH}"
fi

# If clean given, run it prior to any other steps
if [ "$1" == "clean" ]; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
        if [ -d "${bd}" ]; then
            find "${bd}" -mindepth 1 -delete
            rmdir "${bd}" || true
        fi
    done

    # Cleaning up python artifacts
    rm -f "${REPODIR}/legate_raft/install_info.py"
    find ${REPODIR} | grep -E "(__pycache__|\.pyc|\.pyo|\.so|\_skbuild$)"  | xargs rm -rf
    exit 0
fi

# ensure 'native' is used if CUDAARCHS isn't set
# (instead of the CMake default which is a specific architecture)
# ref: https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
declare -r CMAKE_CUDA_ARCHITECTURES="${CUDAARCHS:-native}"

mkdir -p "${BUILD_DIR}"

cmake \
 -B "${BUILD_DIR}" -S "${REPODIR}" \
 -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
 -DRAFT_NVTX=OFF \
 -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
 ${EXTRA_CMAKE_ARGS} \
 .

cmake --build "${BUILD_DIR}" -v -j${PARALLEL_LEVEL}
cmake --build "${BUILD_DIR}" --target install

# Build and install the legate-raft Python package
# TODO: This currently always rebuilds things, which it should not!?
#       and for all I know, it may not be using the quite right config
CMAKE_BUILD_PARALLEL_LEVEL="${PARALLEL_LEVEL:-1}" \
python -m pip install . --no-build-isolation --no-deps --config-settings rapidsai.disable-cuda=true -v
