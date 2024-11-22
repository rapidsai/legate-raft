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
#include "pitches.h"
#include <common/gpu_task_context.hpp>

#include "legate/utilities/dispatch.h"

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <thrust/fill.h>

namespace legate_raft {

namespace {

struct fill_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::PhysicalStore output, legate::Scalar value)
  {
    using VAL = legate::type_of_t<CODE>;

    auto shape = output.shape<DIM>();

    if (shape.empty()) return;

    auto output_acc = output.write_accessor<VAL, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran order*/); it.valid(); ++it) {
      auto p        = *it;
      output_acc[p] = value.value<VAL>();
    }
  }
};

template <typename wo_t, typename value_t, typename shape_t, typename pitches_t>
__global__ void fill_kernel_generic(wo_t out, value_t value, shape_t shape, pitches_t pitches)
{
  int idx     = blockIdx.x * blockDim.x + threadIdx.x;
  auto volume = shape.volume();
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, shape.lo);
  out[point] = value;
}

struct fill_fn_gpu {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::TaskContext& ctx, legate::PhysicalStore output, legate::Scalar value)
  {
    using VAL = legate::type_of_t<CODE>;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    auto shape = output.shape<DIM>();
    if (shape.empty()) return;

    const auto val  = value.value<VAL>();
    auto output_acc = output.write_accessor<VAL, DIM>();
    if (output_acc.accessor.is_dense_row_major(shape)) {
      raft::device_span<VAL> output_span(output_acc.ptr(shape), shape.volume());
      thrust::fill(handle.get_thrust_policy(), output_span.begin(), output_span.end(), val);
    } else {
      Pitches<DIM - 1> pitches;
      auto volume = pitches.flatten(shape);

      int block_size = 256;  // TODO: tune
      int num_blocks = (volume + block_size - 1) / block_size;

      fill_kernel_generic<<<num_blocks, block_size, 0, stream>>>(output_acc, val, shape, pitches);
    }

    handle.sync_stream();
  }
};

}  // namespace

class FillTask : public Task<FillTask, FILL> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto value  = context.scalar(0);
    auto output = context.output(0);

    legate::double_dispatch(
      output.data().dim(), output.data().code(), fill_fn{}, output.data(), value);
  }

  static void gpu_variant(legate::TaskContext context)
  {
    auto value  = context.scalar(0);
    auto output = context.output(0);

    legate::double_dispatch(
      output.data().dim(), output.data().code(), fill_fn_gpu{}, context, output.data(), value);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::FillTask::register_variants();
}

}  // namespace
