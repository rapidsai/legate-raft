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

#include <math.h>

#include "legate_library.h"
#include "legate_raft_cffi.h"

#include <common/gpu_task_context.hpp>

#include "legate/utilities/dispatch.h"

namespace legate_raft {

namespace {

struct power_fn_cpu {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::PhysicalStore output, legate::PhysicalStore input, legate::Scalar value)
  {
    using VAL = legate::type_of_t<CODE>;

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto input_acc  = input.read_accessor<VAL, DIM>();
    auto output_acc = output.write_accessor<VAL, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran order*/); it.valid(); ++it) {
      auto p        = *it;
      output_acc[p] = pow(input_acc[p], value.value<VAL>());
    }
  }
};

template <legate::Type::Code CODE>
constexpr bool is_supported_gpu =
  (CODE == legate::Type::Code::FLOAT32 || CODE == legate::Type::Code::FLOAT64);

template <typename value_t>
__global__ void power_kernel(value_t* out, const value_t* in, value_t value, size_t volume)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = pow(in[idx], value);
}

template __global__ void power_kernel(float*, const float*, float, size_t);
template __global__ void power_kernel(double*, const double*, double, size_t);

struct power_fn_gpu {
  template <legate::Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::PhysicalStore input,
                  legate::Scalar value)
  {
    using VAL = legate::type_of_t<CODE>;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto volume = shape.volume();

    int block_size = 256;  // TODO: tune
    int num_blocks = (volume + block_size - 1) / block_size;

    auto input_acc  = input.read_accessor<VAL, DIM>();
    auto output_acc = output.write_accessor<VAL, DIM>();

    power_kernel<<<num_blocks, block_size, 0, stream>>>(
      output_acc.ptr(shape), input_acc.ptr(shape), value.value<VAL>(), volume);
    handle.sync_stream();
  }

  template <legate::Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<!is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::PhysicalStore input,
                  legate::Scalar value)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

}  // namespace

class PowerTask : public Task<PowerTask, POWER> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto value  = context.scalar(0);
    auto output = context.output(0);

    legate::double_dispatch(
      input.data().dim(), input.data().code(), power_fn_cpu{}, output, input, value);
  }

  static void gpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto value  = context.scalar(0);
    auto output = context.output(0);

    legate::double_dispatch(
      input.data().dim(), input.data().code(), power_fn_gpu{}, context, output, input, value);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::PowerTask::register_variants();
}

}  // namespace
