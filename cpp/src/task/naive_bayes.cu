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
 */
#include "legate.h"

#include "legate_library.h"
#include "legate_raft_cffi.h"

#include <common/gpu_task_context.hpp>

#include "legate/utilities/dispatch.h"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/handle.hpp>
#include <raft/matrix/argmax.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/init.cuh>
#include <thrust/fill.h>

namespace legate_raft {

namespace {

template <legate::Type::Code CODE>
constexpr bool is_supported_gpu =
  (CODE == legate::Type::Code::FLOAT32 || CODE == legate::Type::Code::FLOAT64);

template <typename ro_index_t, typename ro_data_t, typename ro_data_dim2_t, typename wo_data_dim2_t>
__global__ void coo_mm_kernel(ro_index_t A_row,
                              ro_index_t A_col,
                              ro_data_t A_data,
                              ro_data_dim2_t B,
                              wo_data_dim2_t C,
                              uint64_t offset,
                              uint64_t nnz,
                              uint64_t p,
                              bool B_transposed)
{
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) return;
  int64_t i = A_row[tid + offset];
  int64_t k = A_col[tid + offset];
  auto Ax   = A_data[tid + offset];  // {i, k}

  // C_ij = sum A_ik * B_kj  (sum over k={0, n})
  for (int64_t j = 0; j < p; j++) {  // B [n,p]
    auto Bx      = B_transposed ? B[{j, k}] : B[{k, j}];
    auto Cx      = Ax * Bx;
    uint64_t idx = i * p + j;
    atomicAdd(C + idx, Cx);
  }
}

template <typename result_t, typename acc_t>
__global__ void add_kernel(result_t result, acc_t classes, uint64_t volume, uint64_t k)
{
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) return;
  result[tid] += classes[tid % k];
}

template <typename wo_t, typename labels_t, typename ro_t, typename rows_t>
__global__ void invert_labels_kernel(
  wo_t output, labels_t labels, ro_t classes, rows_t A_rows, uint64_t offset, uint64_t volume)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) return;
  output[tid] = classes[labels[tid]];
}

struct naive_bayes_fn_gpu {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore A_row,
                  legate::PhysicalStore A_col,
                  legate::PhysicalStore A_data,
                  legate::PhysicalStore feature_log_prob,
                  legate::PhysicalStore class_log_prior,
                  legate::PhysicalStore classes,
                  legate::PhysicalStore result,
                  int32_t m,
                  int32_t k,
                  int32_t n)
  {
    using VAL = legate::type_of_t<CODE>;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    auto A_row_acc            = A_row.read_accessor<uint64_t, 1>();
    auto A_col_acc            = A_col.read_accessor<uint64_t, 1>();
    auto A_data_acc           = A_data.read_accessor<VAL, 1>();
    auto feature_log_prob_acc = feature_log_prob.read_accessor<VAL, 2>();
    auto class_log_prior_acc  = class_log_prior.read_accessor<VAL, 1>();
    auto classes_acc          = classes.read_accessor<int64_t, 1>();
    auto result_acc           = result.write_accessor<int64_t, 1>();

    // Compute local partition size and offset.
    auto A_row_shape      = A_row.shape<1>();
    const uint64_t offset = A_row_shape.lo[0];
    const uint64_t nnz    = A_row_shape.volume();

    // Operation: tmp = X @ feature_log_prob.T
    auto tmp = raft::make_device_matrix<VAL, uint64_t, raft::row_major>(handle, m, n);
    thrust::fill(handle.get_thrust_policy(), tmp.data_handle(), tmp.data_handle() + tmp.size(), 0);
    int block_size = 256;  // TODO: tune
    int num_blocks = (nnz + block_size - 1) / block_size;
    coo_mm_kernel<<<num_blocks, block_size, 0, stream>>>(A_row_acc,
                                                         A_col_acc,
                                                         A_data_acc,
                                                         feature_log_prob_acc,
                                                         tmp.data_handle(),
                                                         offset,
                                                         nnz,
                                                         n,
                                                         true);

    // Operation: jll = tmp + class_log_prior
    block_size = 256;  // TODO: tune
    num_blocks = (tmp.size() + block_size - 1) / block_size;
    add_kernel<<<num_blocks, block_size, 0, stream>>>(
      tmp.data_handle(), class_log_prior_acc, tmp.size(), n);

    // Operation: indices = argmax(jll, axis=1)
    auto indices  = raft::make_device_vector<VAL, uint64_t>(handle, m);
    auto tmp_view = raft::make_device_matrix_view<const VAL, uint64_t, raft::row_major>(
      tmp.data_handle(), tmp.extent(0), tmp.extent(1));
    raft::matrix::argmax(handle, tmp_view, indices.view());

    // Operation: y = invert_labels(indices, classes)
    auto result_shape = result.shape<1>();
    auto result_view  = raft::make_device_vector_view<int64_t, uint64_t>(
      result_acc.ptr(result_shape), result_shape.volume());
    block_size = 256;  // TODO: tune
    num_blocks = (m + block_size - 1) / block_size;
    invert_labels_kernel<<<num_blocks, block_size, 0, stream>>>(
      result_view.data_handle(), indices.view(), classes_acc, A_row_acc, offset, m);

    handle.sync_stream();
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore A_row,
                  legate::PhysicalStore A_col,
                  legate::PhysicalStore A_data,
                  legate::PhysicalStore feature_log_prob,
                  legate::PhysicalStore class_log_prior,
                  legate::PhysicalStore classes,
                  legate::PhysicalStore result,
                  int32_t m,
                  int32_t k,
                  int32_t n)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

}  // namespace

class NaiveBayesTask : public Task<NaiveBayesTask, NAIVE_BAYES> {
 public:
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void gpu_variant(legate::TaskContext context)
  {
    auto A_row            = context.input(0);
    auto A_col            = context.input(1);
    auto A_data           = context.input(2);
    auto feature_log_prob = context.input(3);
    auto class_log_prior  = context.input(4);
    auto classes          = context.input(5);
    auto result           = context.output(0);

    auto m = context.scalar(0).value<int32_t>();
    auto k = context.scalar(1).value<int32_t>();
    auto n = context.scalar(2).value<int32_t>();

    legate::type_dispatch(A_data.data().code(),
                          naive_bayes_fn_gpu{},
                          context,
                          A_row,
                          A_col,
                          A_data,
                          feature_log_prob,
                          class_log_prior,
                          classes,
                          result,
                          m,
                          k,
                          n);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::NaiveBayesTask::register_variants();
}

}  // namespace
