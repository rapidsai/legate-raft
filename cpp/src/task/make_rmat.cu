/* Copyright 2023 NVIDIA Corporation
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
 *
 */

#include "legate_library.h"
#include "legate_raft_cffi.h"

#include <common/gpu_task_context.hpp>

#include <cuda_runtime.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_span.hpp>
#include <raft/linalg/add.cuh>
#include <raft/random/rmat_rectangular_generator.cuh>
#include <raft/random/rng.cuh>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

namespace legate_raft {

class MakeRmatTask : public Task<MakeRmatTask, MAKE_RMAT> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    legate_raft::GPUTaskContext task_context{context};
    auto handle = task_context.handle();

    auto random_seed = context.scalar(0).value<uint64_t>();
    auto r_scale     = context.scalar(1).value<uint64_t>();
    auto c_scale     = context.scalar(2).value<uint64_t>();

    const auto nnz_max = context.scalar(3).value<uint64_t>();
    const float a      = context.scalar(4).value<float>();
    const float b      = context.scalar(5).value<float>();
    const float c      = context.scalar(6).value<float>();

    auto out_src = context.output(0);
    auto out_dst = context.output(1);

    uint64_t index = context.get_task_index().get_index();
    raft::random::RngState rng_state{random_seed + index};
    auto rows      = raft::make_device_vector<uint64_t>(handle, nnz_max);
    auto cols      = raft::make_device_vector<uint64_t>(handle, nnz_max);
    auto rows_view = raft::make_device_vector_view<uint64_t, uint64_t>(rows.data_handle(), nnz_max);
    auto cols_view = raft::make_device_vector_view<uint64_t, uint64_t>(cols.data_handle(), nnz_max);

    raft::random::rmat_rectangular_gen(
      handle, rng_state, rows_view, cols_view, a, b, c, r_scale, c_scale);

    // Make unique
    auto coo_indices =
      thrust::make_zip_iterator(thrust::make_tuple(rows.data_handle(), cols.data_handle()));
    thrust::sort(handle.get_thrust_policy(), coo_indices, coo_indices + nnz_max);
    auto end = thrust::unique(handle.get_thrust_policy(), coo_indices, coo_indices + nnz_max);
    auto nnz = end - coo_indices;

    // Add offset
    uint64_t offset = index << r_scale;
    raft::linalg::addScalar(
      rows.data_handle(), rows.data_handle(), offset, nnz, handle.get_stream());

    // Copy to output
    auto rows_buf = out_src.data().create_output_buffer<uint64_t, 1>(legate::Point<1>(nnz), true);
    auto rows_buf_view = raft::make_device_vector_view<uint64_t, int32_t>(&rows_buf[0], nnz);
    raft::copy(rows_buf_view.data_handle(), rows.data_handle(), nnz, handle.get_stream());

    auto cols_buf = out_dst.data().create_output_buffer<uint64_t, 1>(legate::Point<1>(nnz), true);
    auto cols_buf_view = raft::make_device_vector_view<uint64_t, int32_t>(&cols_buf[0], nnz);
    raft::copy(cols_buf_view.data_handle(), cols.data_handle(), nnz, handle.get_stream());
  }
};

}  // namespace legate_raft

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  legate_raft::MakeRmatTask::register_variants();
}

}  // namespace
