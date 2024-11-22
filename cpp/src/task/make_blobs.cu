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

#include <optional>
#include <stdexcept>

#include "../legate_library.h"
#include "../legate_raft.h"

#include "common/utils.hpp"
#include <common/gpu_task_context.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/make_blobs.cuh>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>
#include <nccl.h>

namespace legate_raft {

class MAKE_BLOBS_TASK : public Task<MAKE_BLOBS_TASK, MAKE_BLOBS> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    legate_raft::GPUTaskContext task_context{context};

    int n_centers = context.scalar(0).value<int>();

    auto X       = context.output(0).data();
    auto y       = context.output(1).data();
    auto centers = context.input(0).data();

    auto handle = task_context.handle();

    // The offset of the current partition from the start of the store
    uint64_t X_offset = X.shape<2>().lo[0];

    auto X_view = device_matrix_view_from_store<float, int>(X, "X");
    auto y_view = device_vector_view_from_store<int, int>(y, "y");
    // TODO: centers are not passed read-only, so manually convert for now:
    int n_features = centers.shape<2>().hi[1] + 1;
    float* centers_read =
      const_cast<float*>(centers.read_accessor<float, 2>().ptr(centers.shape<2>()));
    auto centers_view =
      raft::make_device_matrix_view<float, int>(centers_read, n_centers, n_features);

    raft::random::make_blobs<float, int>(handle,
                                         X_view,
                                         y_view,
                                         n_centers,
                                         std::optional(centers_view),
                                         std::nullopt,
                                         true,
                                         -10.0,
                                         10.0,
                                         X_offset);
  }
};

}  // namespace legate_raft

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  legate_raft::MAKE_BLOBS_TASK::register_variants();
}

}  // namespace
