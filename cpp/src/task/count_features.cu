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

#include <cstdint>
#include <iostream>  // TODO: remove after debugging

#include <common/gpu_task_context.hpp>

#include "legate/utilities/dispatch.h"

#include "../legate_raft.h"
#include "../raft/raft_api.hpp"
#include "legate_library.h"

namespace legate_raft {

namespace {

template <legate::Type::Code CODE>
constexpr bool is_supported = legate::is_floating_point<CODE>::value;

struct sparse_count_features_fn_cpu {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(legate::PhysicalStore data,
                  legate::PhysicalStore rows,
                  legate::PhysicalStore cols,
                  legate::PhysicalStore labels,
                  legate::PhysicalStore result)
  {
    using VAL = legate::type_of_t<CODE>;

    auto shape = data.shape<1>();

    auto data_acc = data.read_accessor<VAL, 1>();
    auto rows_acc = rows.read_accessor<int32_t, 1>();
    auto cols_acc = cols.read_accessor<int32_t, 1>();

    auto labels_acc = labels.read_accessor<int64_t, 1>();
    auto result_acc = result.reduce_accessor<legate::SumReduction<VAL>, true, 2>();

    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      auto p     = *it;
      auto value = data_acc[p];
      auto row   = rows_acc[p];
      auto col   = cols_acc[p];
      auto label = labels_acc[row];

      result_acc.reduce(legate::Point<2>(label, col), value);
    }
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported<CODE>>* = nullptr>
  void operator()(legate::PhysicalStore data,
                  legate::PhysicalStore rows,
                  legate::PhysicalStore cols,
                  legate::PhysicalStore labels,
                  legate::PhysicalStore result)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

template <typename out_t, typename value_t, typename index_t, typename label_t>
__global__ void count_features_coo_kernel(out_t out,
                                          index_t* rows,
                                          index_t* cols,
                                          const value_t* vals,
                                          int nnz,
                                          int n_rows,
                                          int n_cols,
                                          const label_t* labels,
                                          // value_t *weights,
                                          // bool has_weights,
                                          int n_features,
                                          bool square)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nnz) return;

  index_t row   = rows[i];
  index_t col   = cols[i];
  value_t val   = vals[i];
  label_t label = labels[row];

  auto out_idx = (label * n_features) + col;

  // if (has_weights) val *= weights[i];
  if (square) val *= val;
  legate::Point<2> pos_reduce(static_cast<int64_t>(label), static_cast<int64_t>(col));
  out.reduce(pos_reduce, val);
  // atomicAdd(out + out_idx, val);
}

template <legate::Type::Code CODE>
constexpr bool is_supported_gpu =
  (CODE == legate::Type::Code::FLOAT32 || CODE == legate::Type::Code::FLOAT64);

struct sparse_count_features_fn_gpu {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore data,
                  legate::PhysicalStore rows,
                  legate::PhysicalStore cols,
                  legate::PhysicalStore labels,
                  legate::PhysicalStore result,
                  uint64_t n_rows,
                  uint64_t n_cols,
                  uint64_t n_features)
  {
    using value_t = legate::type_of_t<CODE>;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    auto shape = data.shape<1>();

    auto data_acc   = data.read_accessor<value_t, 1>();
    auto rows_acc   = rows.read_accessor<uint64_t, 1>();
    auto cols_acc   = cols.read_accessor<uint64_t, 1>();
    auto labels_acc = labels.read_accessor<int64_t, 1>();
    auto result_acc = result.reduce_accessor<legate::SumReduction<value_t>, false, 2>();

    auto data_shape = data.shape<1>();
    auto offset     = data_shape.lo[0];
    auto nnz        = data_shape.hi[0] + 1 - offset;

    int block_size = 256;  // TODO: tune
    int num_blocks = (nnz + block_size - 1) / block_size;

    count_features_coo_kernel<<<num_blocks, block_size, 0, stream>>>(result_acc,
                                                                     // result_acc.ptr({0, 0}),
                                                                     rows_acc.ptr(offset),
                                                                     cols_acc.ptr(offset),
                                                                     data_acc.ptr(offset),
                                                                     nnz,
                                                                     n_rows,
                                                                     n_cols,
                                                                     labels_acc.ptr(0),
                                                                     // weights, has_weights,
                                                                     n_features,
                                                                     false);
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore data,
                  legate::PhysicalStore rows,
                  legate::PhysicalStore cols,
                  legate::PhysicalStore labels,
                  legate::PhysicalStore result,
                  uint64_t n_rows,
                  uint64_t n_cols,
                  uint64_t n_features)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

}  // namespace

class SparseCountFeaturesTask : public Task<SparseCountFeaturesTask, COUNT_FEATURES> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto X_data = context.input(0);
    auto X_rows = context.input(1);
    auto X_cols = context.input(2);

    auto labels = context.input(3);
    auto result = context.reduction(0);

    legate::type_dispatch(
      X_data.data().code(), sparse_count_features_fn_cpu{}, X_data, X_rows, X_cols, labels, result);
  }

  static void gpu_variant(legate::TaskContext context)
  {
    auto X_data = context.input(0);
    auto X_rows = context.input(1);
    auto X_cols = context.input(2);

    auto labels = context.input(3);

    auto n_rows     = context.scalar(1).value<uint64_t>();
    auto n_cols     = context.scalar(2).value<uint64_t>();
    auto n_features = context.scalar(3).value<uint64_t>();

    auto result = context.reduction(0);

    legate::type_dispatch(X_data.data().code(),
                          sparse_count_features_fn_gpu{},
                          context,
                          X_data,
                          X_rows,
                          X_cols,
                          labels,
                          result,
                          n_rows,
                          n_cols,
                          n_features);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::SparseCountFeaturesTask::register_variants();
}

}  // namespace
