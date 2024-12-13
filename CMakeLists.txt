# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

set(RAPIDS_VERSION "24.10")

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project -------------- #

# read project version from VERSION file
file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" _version_content)
if(_version_content MATCHES [[^([0-9]+)\.([0-9]+)\.([0-9]+)]])
  set(_legateraft_version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
else()
  string(REPLACE "\n" "\n  " _legateraft_content_formatted "  ${_version_content}")
  message(
    FATAL_ERROR
      "Could not determine project version. Contents of VERSION file:\n${_legateraft_content_formatted}"
  )
endif()

rapids_cuda_init_architectures(legate_raft)
project(legate_raft VERSION "${_legateraft_version}" LANGUAGES C CXX CUDA)

# ------------- configure raft ----------------- #

rapids_cpm_init()
# Need to find UCX, related to https://github.com/rapidsai/ucxx/issues/173
# and others used by dask distributed.
# Needed as of RAPIDS 24.06 if raft needs to be build
# (if not, we may need some or all of these anyway though).
find_package(ucx REQUIRED)
find_package(ucxx REQUIRED)
rapids_find_generate_module(
  NCCL
  HEADER_NAMES nccl.h
  LIBRARY_NAMES nccl
)
find_package(NCCL REQUIRED)

include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_raft.cmake)

# -------------- add requirements -------------- #

find_package(legate REQUIRED)
find_package(LegateDataframe REQUIRED)
set(BUILD_SHARED_LIBS ON)

# -------------- compile tasks ----------------- #

# C++ layer
add_subdirectory(cpp/src)

include(cmake/legate_raft/generate_install_info.cmake)
legate_raft_generate_install_info(
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/legate_raft_cffi.h
  TARGET legate_raft
  PY_PATH legate_raft)

include(GNUInstallDirs)

rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_raft DESTINATION ${lib_dir} EXPORT legate_raft-export)

install(DIRECTORY cmake/legate_raft/
        DESTINATION "${lib_dir}/cmake/legate_raft" FILES_MATCHING
        PATTERN "*.cmake")

rapids_export(INSTALL legate_raft
  EXPORT_SET legate_raft-export
  GLOBAL_TARGETS legate_raft
  NAMESPACE legate::)

# build export targets
rapids_export(BUILD legate_raft
  EXPORT_SET legate_raft-export
  GLOBAL_TARGETS legate_raft
  NAMESPACE legate::)

if(SKBUILD)
  add_library(legate_raft_python INTERFACE)
  add_library(legate::legate_raft_python ALIAS legate_raft_python)
  target_link_libraries(legate_raft_python INTERFACE legate::legate legate::legate_raft)

  install(TARGETS legate_raft_python DESTINATION ${lib_dir} EXPORT legate_raft-export)

  rapids_export(INSTALL legate_raft_python EXPORT_SET legate_raft-export
    GLOBAL_TARGETS legate_raft_python NAMESPACE legate::)
endif()
