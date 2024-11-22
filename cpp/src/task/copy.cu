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
#include "legate/utilities/typedefs.h"

namespace legate_raft {

namespace {

struct copy_fn_cpu {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::PhysicalStore output, legate::PhysicalStore input)
  {
    using VAL = legate::type_of_t<CODE>;

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto input_acc  = input.read_accessor<VAL, DIM>();
    auto output_acc = output.write_accessor<VAL, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran order*/); it.valid(); ++it) {
      auto p        = *it;
      output_acc[p] = input_acc[p];
    }
  }
};

template <typename accessor_wo, typename accessor_ro, typename shape_t, typename pitches_t>
__global__ void copy_kernel(accessor_wo out, accessor_ro in, shape_t shape, pitches_t pitches)
{
  int idx     = blockIdx.x * blockDim.x + threadIdx.x;
  auto volume = shape.volume();
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, shape.lo);
  out[point] = in[point];
}

struct copy_fn_gpu {
  template <legate::Type::Code CODE, int32_t DIM>
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

    auto input_acc  = input.read_accessor<VAL, DIM>();
    auto output_acc = output.write_accessor<VAL, DIM>();

    Pitches<DIM - 1> pitches;
    auto volume = pitches.flatten(shape);

    const int block_size = 256;  // TODO: tune
    const int num_blocks = (volume + block_size - 1) / block_size;

    copy_kernel<<<num_blocks, block_size, 0, stream>>>(output_acc, input_acc, shape, pitches);
    handle.sync_stream();
  }
};

}  // namespace

class CopyTask : public Task<CopyTask, COPY> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto output = context.output(0);

    legate::double_dispatch(input.data().dim(), input.data().code(), copy_fn_cpu{}, output, input);
  }

  static void gpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto output = context.output(0);

    legate::double_dispatch(
      input.data().dim(), input.data().code(), copy_fn_gpu{}, context, output, input);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::CopyTask::register_variants();
}

}  // namespace
