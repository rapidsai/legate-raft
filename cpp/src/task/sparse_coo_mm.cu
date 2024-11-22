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
#include <raft/matrix/copy.cuh>
#include <raft/matrix/init.cuh>

namespace legate_raft {

namespace {

// struct sparse_coo_mm_fn_cpu {

//   template <legate::Type::Code CODE>
//   void operator()(
//       legate::PhysicalStore A_data, legate::PhysicalStore A_row, legate::PhysicalStore Ap,
//       legate::PhysicalStore B, legate::PhysicalStore C)
//   {
//     using VAL = legate::type_of_t<CODE>;

//     // Remove promotion that was only needed for alignment.
//     if (Ap.transformed()) { Ap.remove_transform(); }

//     auto A_data_acc = A_data.read_accessor<VAL, 1>();
//     auto A_row_acc = A_row.read_accessor<int32_t, 1>();
//     auto Ap_acc = Ap.read_accessor<legate::Rect<1>, 1>();

//     const auto shape_ = C.shape<2>();

//     auto p = B.shape<2>().hi[1] + 1;
//     auto B_acc = B.read_accessor<VAL, 2>();
//     auto C_acc = C.reduce_accessor<legate::SumReduction<VAL>, true, 2>();

//     for (auto i = shape_.lo[0]; i < shape_.hi[0] + 1; ++i) {
//       for (int32_t jj = Ap_acc[i].lo; jj < Ap_acc[i].hi; ++jj) {
//         auto j = A_row_acc[jj];
//         auto A_val = A_data_acc[jj];
//         for (int32_t k = 0; k < p; ++k) {
//           auto B_val = B_acc[{j, k}];
//           C_acc.reduce({i, k}, A_val * B_val);
//         }
//       }
//     }
//   }
// };

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
                              uint64_t n)
{
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) return;
  int64_t i = A_row[tid + offset];
  int64_t k = A_col[tid + offset];
  auto Ax   = A_data[tid + offset];  // {i, k}
  for (int64_t j = 0; j < n; j++) {
    auto Cx = Ax * B[{k, j}];
  }
}

struct sparse_coo_mm_fn_gpu {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore A_row,
                  legate::PhysicalStore A_col,
                  legate::PhysicalStore A_data,
                  legate::PhysicalStore B,
                  legate::PhysicalStore C,
                  int32_t m,
                  int32_t k,
                  int32_t n,
                  uint64_t nnz)
  {
    using VAL = legate::type_of_t<CODE>;

    // Remove promotion that was only needed for alignment.
    if (A_row.transformed()) { A_row.remove_transform(); }
    if (A_col.transformed()) { A_col.remove_transform(); }
    if (A_data.transformed()) { A_data.remove_transform(); }

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    auto A_row_acc  = A_row.read_accessor<uint64_t, 1>();
    auto A_col_acc  = A_col.read_accessor<uint64_t, 1>();
    auto A_data_acc = A_data.read_accessor<VAL, 1>();

    auto B_acc = B.read_accessor<VAL, 2>();
    auto C_acc = C.reduce_accessor<legate::SumReduction<VAL>, false, 2>();

    // Compute local partition size and offset.
    auto A_row_shape       = A_row.shape<1>();
    const uint64_t offset  = A_row_shape.lo[0];
    const uint64_t local_m = A_row_shape.volume();
    std::cerr << "A_row_shape: " << A_row_shape << " offset=" << offset << " local_nzz=" << local_m
              << "\n";

    auto local_C = raft::make_device_matrix<VAL, int32_t>(handle, local_m, n);

    int block_size = 256;  // TODO: tune
    int num_blocks = (nnz + block_size - 1) / block_size;
    coo_mm_kernel<<<num_blocks, block_size, 0, stream>>>(
      A_row_acc, A_col_acc, A_data_acc, B_acc, C_acc, offset, nnz, n);
    handle.sync_stream();
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::PhysicalStore A_data,
                  legate::PhysicalStore A_row,
                  legate::PhysicalStore Ap,
                  legate::PhysicalStore B,
                  legate::PhysicalStore C,
                  int32_t m,
                  int32_t k,
                  int32_t n,
                  uint64_t nnz)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

}  // namespace

class SparseCOOMMTask : public Task<SparseCOOMMTask, SPARSE_COO_MM> {
 public:
  // static void cpu_variant(legate::TaskContext context)
  // {
  //   auto Ap = context.input(0);
  //   auto A_row = context.input(1);
  //   auto A_data = context.input(2);
  //   auto B = context.input(3);
  //   auto C = context.reduction(0);

  //   legate::type_dispatch(A_data.data().code(), sparse_coo_mm_fn_cpu{}, A_data, A_row, Ap, B, C);
  // }

  static void gpu_variant(legate::TaskContext context)
  {
    auto A_row  = context.input(0);
    auto A_col  = context.input(1);
    auto A_data = context.input(2);
    auto B      = context.input(3);
    auto C      = context.reduction(0);
    // auto C = context.output(0);

    auto m   = context.scalar(0).value<int32_t>();
    auto k   = context.scalar(1).value<int32_t>();
    auto n   = context.scalar(2).value<int32_t>();
    auto nnz = context.scalar(3).value<uint64_t>();

    legate::type_dispatch(A_data.data().code(),
                          sparse_coo_mm_fn_gpu{},
                          context,
                          A_row,
                          A_col,
                          A_data,
                          B,
                          C,
                          m,
                          k,
                          n,
                          nnz);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::SparseCOOMMTask::register_variants();
}

}  // namespace
