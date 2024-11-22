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

#include <cmath>

#include "legate_library.h"
#include "legate_raft_cffi.h"
#include "pitches.h"

#include <common/gpu_task_context.hpp>

#include "legate/utilities/dispatch.h"

#include <raft/core/handle.hpp>

namespace legate_raft {

namespace {

struct add_fn_cpu {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::PhysicalStore output, legate::PhysicalStore x1, legate::PhysicalStore x2)
  {
    using VAL = legate::type_of_t<CODE>;

    auto shape = x1.shape<DIM>();

    if (shape.empty()) return;

    auto x1_acc     = x1.read_accessor<VAL, DIM>();
    auto x2_acc     = x2.read_accessor<VAL, DIM>();
    auto output_acc = output.write_accessor<VAL, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran order*/); it.valid(); ++it) {
      auto p        = *it;
      output_acc[p] = x1_acc[p] + x2_acc[p];
    }
  }
};

template <legate::Type::Code CODE>
constexpr bool is_supported_gpu =
  (CODE == legate::Type::Code::FLOAT32 || CODE == legate::Type::Code::FLOAT64);

template <typename value_t>
__global__ void add_kernel_dense(value_t* out, const value_t* x1, const value_t* x2, int volume)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = x1[idx] + x2[idx];
}

template <typename wo_t, typename ro_t, typename shape_t, typename pitches_t>
__global__ void add_kernel_generic(wo_t out, ro_t x1, ro_t x2, shape_t shape, pitches_t pitches)
{
  int idx     = blockIdx.x * blockDim.x + threadIdx.x;
  auto volume = shape.volume();
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, shape.lo);
  out[point] = x1[point] + x2[point];
}

struct add_fn_gpu {
  template <legate::Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::PhysicalStore x1,
                  legate::PhysicalStore x2)
  {
    using VAL = legate::type_of_t<CODE>;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();
    auto shape  = x1.shape<DIM>();

    if (shape.empty()) return;

    auto x1_acc     = x1.read_accessor<VAL, DIM>();
    auto x2_acc     = x2.read_accessor<VAL, DIM>();
    auto output_acc = output.write_accessor<VAL, DIM>();

    Pitches<DIM - 1> pitches;
    auto volume = pitches.flatten(shape);

    int block_size = 256;  // TODO: tune
    int num_blocks = (volume + block_size - 1) / block_size;

    bool dense =
      x1_acc.accessor.is_dense_row_major(shape) && x2_acc.accessor.is_dense_row_major(shape);

    if (dense) {
      add_kernel_dense<<<num_blocks, block_size, 0, stream>>>(
        output_acc.ptr(shape), x1_acc.ptr(shape), x2_acc.ptr(shape), volume);
    } else {
      add_kernel_generic<<<num_blocks, block_size, 0, stream>>>(
        output_acc, x1_acc, x2_acc, shape, pitches);
    }
    handle.sync_stream();
  }

  template <legate::Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<!is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::PhysicalStore x1,
                  legate::PhysicalStore x2)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

}  // namespace

class AddTask : public Task<AddTask, ADD> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto input1 = context.input(0);
    auto input2 = context.input(1);
    auto output = context.output(0);

    legate::double_dispatch(input1.data().dim(),
                            input1.data().code(),
                            add_fn_cpu{},
                            output.data(),
                            input1.data(),
                            input2.data());
  }

  static void gpu_variant(legate::TaskContext context)
  {
    auto input1 = context.input(0);
    auto input2 = context.input(1);
    auto output = context.output(0);

    legate::double_dispatch(input1.data().dim(),
                            input1.data().code(),
                            add_fn_gpu{},
                            context,
                            output.data(),
                            input1.data(),
                            input2.data());
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::AddTask::register_variants();
}

}  // namespace
