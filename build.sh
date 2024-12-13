#!/bin/bash

# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# raft empty project template build script

# Abort script on first error
set -e

INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}

PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}

BUILD_TYPE=Release
BUILD_DIR=build/

RAFT_REPO_REL=${RAFT_REPO_REL:-}
EXTRA_CMAKE_ARGS=${EXTRA_CMAKE_ARGS:-}


if [[ ${RAFT_REPO_REL} != "" ]]; then
  RAFT_REPO_PATH="`readlink -f \"${RAFT_REPO_REL}\"`"
  EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCPM_raft_SOURCE=${RAFT_REPO_PATH}"
fi

if [ "$1" == "clean" ]; then
  rm -rf cpp/build
  rm -rf dist legate_raft.egg-info
  python setup.py clean --all
  rm -f legate_raft/install_info.py
  rm -rf pytest/__pycache__
  exit 0
fi

# ensure 'native' is used if CUDAARCHS isn't set
# (instead of the CMake default which is a specific architecture)
# ref: https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
declare -r CMAKE_CUDA_ARCHITECTURES="${CUDAARCHS:-native}"

mkdir -p cpp/$BUILD_DIR
cd cpp/$BUILD_DIR

cmake  \
 -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
 -DRAFT_NVTX=OFF \
 -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
 ${EXTRA_CMAKE_ARGS} \
 ../../

cmake --build . -v -j${PARALLEL_LEVEL}
cmake --install . --prefix ${INSTALL_PREFIX}

cd ../..
python -m pip install . --no-build-isolation --no-deps --config-settings rapidsai.disable-cuda=true
