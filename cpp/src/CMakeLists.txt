# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(
        legate_raft

        # RAFT API Wrappers
        raft/distance.cu
        common/gpu_task_context.cu
        raft/raft_kmeans.cu

        # Legate tasks
        task/add.cu
        task/add_constant.cu
        task/bincount.cu
        task/convert.cu
        task/count_features.cu
        task/fill.cu
        task/find_max.cc
        task/hashing_vectorizer.cu
        task/invert_labels.cu
        task/kmeans_tasks.cu
        task/knn_task.cu
        task/make_blobs.cu
        task/log.cu
        task/make_rmat.cu
        task/map_labels.cu
        task/mul.cc
        task/multiply_by_constant.cu
        task/naive_bayes.cu
        task/power.cu
        task/range.cc
        task/rng.cu
        task/sum_over_axis.cu
        task/tfidf.cu
        task/unique.cu

        # Library templates
        legate_library.cc
        mapper.cc
)

target_include_directories(legate_raft
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        INTERFACE
        $<INSTALL_INTERFACE:include>
        )

target_link_libraries(legate_raft PRIVATE legate::legate LegateDataframe::LegateDataframe raft::raft raft::compiled raft::distributed)
