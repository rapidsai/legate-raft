/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdexcept>

#include "legate.h"

#include "../legate_raft.h"

#include "common/gpu_task_context.hpp"
#include "common/utils.hpp"

#include "../raft/raft_kmeans_api.hpp"

#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>
#include <nccl.h>

namespace legate_raft {

namespace {

template <typename DataT>
static void fit_impl(raft::handle_t const& handle,
                     legate::PhysicalStore X,
                     legate::PhysicalStore centroids,
                     legate::PhysicalStore info)
{
  int rank = handle.comms_initialized() ? handle.get_comms().get_rank() : 0;

  auto X_view         = device_matrix_view_from_store<const DataT, int>(X, "X");
  auto centroids_view = device_matrix_view_from_store<DataT, int>(centroids, "centroids");

  raft::cluster::KMeansParams params;
  params.n_clusters = centroids_view.extent(0);

  /* Algorithm information results: */
  DataT inertia = 0;  // Never found currently (could be used as stopping criterion).
  int n_iter;

  if (!handle.comms_initialized()) {
    /* The task is running on a single GPU, so use raft directly. */
    raft::cluster::kmeans::fit<DataT, int>(handle,
                                           params,
                                           X_view,
                                           std::nullopt,
                                           centroids_view,
                                           raft::make_host_scalar_view(&inertia),
                                           raft::make_host_scalar_view(&n_iter));
  } else {
    /* Otherwise call our custom multi-node multi-gpu version. */
    kmeans::fit<DataT, int>(handle,
                            params,
                            X_view,
                            std::nullopt,
                            centroids_view,
                            raft::make_host_scalar_view(&inertia),
                            raft::make_host_scalar_view(&n_iter));
  }

  if (rank == 0) {
    /*
     * TODO: This binds GPU data which makes the copy annoying.  A mapper could
     *       fix this.  Z_COPY_MEM might also work, or this might be scalar?
     */
    auto info_buf = info.create_output_buffer<DataT, 2>({1, 1}, true);
    DataT vals[1] = {(DataT)n_iter};
    raft::copy(info_buf.ptr({0, 0}), vals, sizeof(vals), handle.get_stream());
  } else {
    info.bind_empty_data();
  }
}

}  // namespace

class RAFT_KMEANS_FIT_TASK : public Task<RAFT_KMEANS_FIT_TASK, RAFT_KMEANS_FIT> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    legate_raft::GPUTaskContext task_context{context};

    auto X         = context.input(0).data();
    auto centroids = context.output(0).data();
    auto info      = context.output(1).data();

    if (!context.is_single_task()) {
      ncclComm_t nccl_com = *(context.communicators().at(0).get<ncclComm_t*>());
      task_context.inject_nccl_comm(nccl_com);
    }

    auto handle = task_context.handle();

    if (X.code() == legate::Type::Code::FLOAT32) {
      fit_impl<float>(handle, X, centroids, info);
    } else if (X.code() == legate::Type::Code::FLOAT64) {
      fit_impl<double>(handle, X, centroids, info);
    } else {
      throw std::invalid_argument("X must be float32 or float64.");
    }
  }
};

namespace  // unnamed
{

template <typename DataT>
static void predict_impl(raft::handle_t handle,
                         legate::PhysicalStore X,
                         legate::PhysicalStore centroids,
                         legate::PhysicalStore labels)
{
  auto X_view         = device_matrix_view_from_store<const DataT, int>(X, "X");
  auto centroids_view = device_matrix_view_from_store<const DataT, int>(centroids, "centroids");
  // The templated 2 means labels is 2-D but only the first dim is used.
  auto labels_view = device_vector_view_from_store<int, int, 2>(labels, "labels");

  raft::cluster::KMeansParams params;
  params.n_clusters = centroids_view.extent(0);
  DataT inertia;
  raft::cluster::kmeans::predict<DataT, int>(handle,
                                             params,
                                             X_view,
                                             std::nullopt,
                                             centroids_view,
                                             labels_view,
                                             false,
                                             raft::make_host_scalar_view(&inertia));
}

}  // namespace

class RAFT_KMEANS_PREDICT_TASK : public Task<RAFT_KMEANS_PREDICT_TASK, RAFT_KMEANS_PREDICT> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    legate_raft::GPUTaskContext task_context{context};

    auto X         = context.input(0).data();
    auto centroids = context.input(1).data();
    auto labels    = context.output(0).data();

    auto handle = task_context.handle();

    if (X.code() == legate::Type::Code::FLOAT32) {
      predict_impl<float>(handle, X, centroids, labels);
    } else if (X.code() == legate::Type::Code::FLOAT64) {
      predict_impl<double>(handle, X, centroids, labels);
    } else {
      throw std::invalid_argument("X must be float32 or float64.");
    }
  }
};

}  // namespace legate_raft

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  legate_raft::RAFT_KMEANS_FIT_TASK::register_variants();
  legate_raft::RAFT_KMEANS_PREDICT_TASK::register_variants();
}

}  // namespace
