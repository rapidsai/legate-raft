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

#include "pitches.h"

#include "legate/utilities/dispatch.h"

#include <raft/core/handle.hpp>

namespace legate_raft {

namespace {

template <typename rd_t, typename ro_t, typename shape_t>
__global__ void bincount_kernel(rd_t out, ro_t in, shape_t in_shape, shape_t out_shape)
{
  int idx     = blockIdx.x * blockDim.x + threadIdx.x;
  auto volume = in_shape.volume();
  if (idx >= volume) return;
  auto value = in[in_shape.lo + idx];
  legate::Point<1> pos_reduce(static_cast<int64_t>(value));
  if (out_shape.contains(pos_reduce)) out.reduce(pos_reduce, 1);
}

template <legate::Type::Code SRC_TYPE>
struct bincount_fn_gpu {
  template <legate::Type::Code DST_TYPE>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore input,
                  legate::PhysicalStore output)
  {
    using SRC = legate::type_of_t<SRC_TYPE>;
    using DST = legate::type_of_t<DST_TYPE>;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    auto in_shape  = input.shape<1>();
    auto out_shape = output.shape<1>();

    auto in_acc  = input.read_accessor<SRC, 1>();
    auto out_acc = output.reduce_accessor<legate::SumReduction<DST>, false, 1>();

    auto volume    = in_shape.volume();
    int block_size = 256;  // TODO: tune
    int num_blocks = (volume + block_size - 1) / block_size;

    bincount_kernel<<<num_blocks, block_size, 0, stream>>>(out_acc, in_acc, in_shape, out_shape);
    handle.sync_stream();
  }
};

template <legate::Type::Code SRC_TYPE>
constexpr bool is_supported =
  (legate::is_integral<SRC_TYPE>::value && SRC_TYPE != legate::Type::Code::BOOL);

struct SourceTypeDispatchGPU {
  template <legate::Type::Code SRC_TYPE, std::enable_if_t<is_supported<SRC_TYPE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore input,
                  legate::PhysicalStore output)
  {
    legate::type_dispatch(output.code(), bincount_fn_gpu<SRC_TYPE>{}, ctx, input, output);
  }

  template <legate::Type::Code SRC_TYPE, std::enable_if_t<!is_supported<SRC_TYPE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore input,
                  legate::PhysicalStore output)
  {
    LEGATE_ABORT("Source type not supported");
  }
};

template <legate::Type::Code SRC_TYPE>
struct bincount_fn_cpu {
  template <legate::Type::Code DST_TYPE>
  void operator()(legate::PhysicalStore input, legate::PhysicalStore output)
  {
    using SRC = legate::type_of_t<SRC_TYPE>;
    using DST = legate::type_of_t<DST_TYPE>;

    auto in_shape  = input.shape<1>();
    auto out_shape = output.shape<1>();

    auto in_acc  = input.read_accessor<SRC, 1>();
    auto out_acc = output.reduce_accessor<legate::SumReduction<DST>, true, 1>();

    for (legate::PointInRectIterator<1> it(in_shape); it.valid(); ++it) {
      auto& value = in_acc[*it];
      legate::Point<1> pos_reduce(static_cast<int64_t>(value));

      if (out_shape.contains(pos_reduce)) out_acc.reduce(pos_reduce, 1);
    }
  }
};

struct SourceTypeDispatchCPU {
  template <legate::Type::Code SRC_TYPE, std::enable_if_t<is_supported<SRC_TYPE>>* = nullptr>
  void operator()(legate::PhysicalStore input, legate::PhysicalStore output)
  {
    legate::type_dispatch(output.code(), bincount_fn_cpu<SRC_TYPE>{}, input, output);
  }

  template <legate::Type::Code SRC_TYPE, std::enable_if_t<!is_supported<SRC_TYPE>>* = nullptr>
  void operator()(legate::PhysicalStore input, legate::PhysicalStore output)
  {
    LEGATE_ABORT("Source type not supported");
  }
};

}  // namespace

class BincountTask : public Task<BincountTask, BINCOUNT> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto output = context.reduction(0);

    legate::type_dispatch(input.data().code(), SourceTypeDispatchGPU{}, context, input, output);
  }

  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto output = context.reduction(0);

    legate::type_dispatch(input.data().code(), SourceTypeDispatchCPU{}, input, output);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::BincountTask::register_variants();
}

}  // namespace
