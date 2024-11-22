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

#include "legate/utilities/dispatch.h"
#include "legate/utilities/typedefs.h"
#include "legate_library.h"
#include "legate_raft_cffi.h"
#include "pitches.h"
#include <common/gpu_task_context.hpp>

#include <raft/core/handle.hpp>

namespace legate_raft {

namespace {

struct reduction_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::PhysicalStore output, legate::PhysicalStore input)
  {
    using VAL = legate::type_of_t<CODE>;

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto in_acc  = input.read_accessor<VAL, DIM>();
    auto red_acc = output.reduce_accessor<legate::SumReduction<VAL>, true, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran_order*/); it.valid(); ++it) {
      auto p = *it;
      // Coordinates of the contracting dimension are ignored by red_acc via an affine
      // transformation. For example, if the store was 3D and the second dimension was contracted,
      // each point p will go through the following affine transformation to recover the point in
      // the domain prior to the promotion:
      //
      //     | 1  0  0 |     | x |
      //     |         |  *  | y |
      //     | 0  0  1 |     | z |
      //
      // where the "*" operator denotes a matrix-vector multiplication.
      red_acc.reduce(p, in_acc[p]);
    }
  }
};

template <typename rd_t, typename ro_t, typename shape_t, typename pitches_t>
__global__ void sum_over_axis_kernel(rd_t out, ro_t in, shape_t shape, pitches_t pitches)
{
  int idx     = blockIdx.x * blockDim.x + threadIdx.x;
  auto volume = shape.volume();
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, shape.lo);
  out.reduce(point, in[point]);
}

template <legate::Type::Code CODE>
constexpr bool is_supported =
  (CODE == legate::Type::Code::FLOAT32 || CODE == legate::Type::Code::FLOAT64);

struct reduction_fn_gpu {
  template <legate::Type::Code CODE, int32_t DIM, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::PhysicalStore input)
  {
    using VAL = legate::type_of_t<CODE>;
    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto in_acc  = input.read_accessor<VAL, DIM>();
    auto red_acc = output.reduce_accessor<legate::SumReduction<VAL>, false, DIM>();

    Pitches<DIM - 1> pitches;
    auto volume = pitches.flatten(shape);

    const int block_size  = 256;  // TODO: tune
    const auto num_blocks = (volume + block_size - 1) / block_size;

    sum_over_axis_kernel<<<num_blocks, block_size, 0, stream>>>(red_acc, in_acc, shape, pitches);
    handle.sync_stream();
  }

  template <legate::Type::Code CODE, int32_t DIM, std::enable_if_t<!is_supported<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::PhysicalStore input)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

}  // namespace

class SumOverAxisTask : public Task<SumOverAxisTask, SUM_OVER_AXIS> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto output = context.reduction(0);

    legate::double_dispatch(input.data().dim(), input.data().code(), reduction_fn{}, output, input);
  }

  static void gpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto output = context.reduction(0);

    legate::double_dispatch(
      input.data().dim(), input.data().code(), reduction_fn_gpu{}, context, output, input);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::SumOverAxisTask::register_variants();
}

}  // namespace
