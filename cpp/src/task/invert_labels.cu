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

#include "legate/utilities/dispatch.h"
#include "legate/utilities/typedefs.h"

namespace legate_raft {

namespace {

template <legate::Type::Code CODE>
constexpr bool is_supported =
  (legate::is_integral<CODE>::value && CODE != legate::Type::Code::BOOL);

struct invert_labels_fn {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(legate::PhysicalStore labels,
                  legate::PhysicalStore classes,
                  legate::PhysicalStore output)
  {
    using VAL = legate::type_of_t<CODE>;

    const auto shape = labels.shape<1>();

    auto labels_acc  = labels.read_accessor<VAL, 1>();
    auto classes_acc = classes.read_accessor<VAL, 1>();
    auto output_acc  = output.write_accessor<VAL, 1>();

    for (legate::PointInRectIterator<1> it(shape, false); it.valid(); ++it) {
      auto idx          = labels_acc[*it];
      auto mapped_label = classes_acc[idx];
      output_acc[*it]   = mapped_label;
    }
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported<CODE>>* = nullptr>
  void operator()(legate::PhysicalStore labels,
                  legate::PhysicalStore classes,
                  legate::PhysicalStore output)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

template <typename wo_t, typename ro_t>
__global__ void invert_labels_kernel(
  wo_t output, ro_t labels, ro_t classes, size_t offset, size_t volume)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) return;
  auto idx    = tid + offset;
  output[idx] = classes[labels[idx]];
}

struct invert_labels_fn_gpu {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore labels,
                  legate::PhysicalStore classes,
                  legate::PhysicalStore output)
  {
    using VAL = legate::type_of_t<CODE>;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto stream = gpu_task_context.handle().get_stream();

    const auto shape = labels.shape<1>();

    auto labels_acc  = labels.read_accessor<VAL, 1>();
    auto classes_acc = classes.read_accessor<VAL, 1>();
    auto output_acc  = output.write_accessor<VAL, 1>();

    size_t offset = shape.lo[0];
    size_t volume = shape.volume();

    int block_size = 256;
    int num_blocks = (volume + block_size - 1) / block_size;
    invert_labels_kernel<<<num_blocks, block_size, 0, stream>>>(
      output_acc, labels_acc, classes_acc, offset, volume);
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore labels,
                  legate::PhysicalStore classes,
                  legate::PhysicalStore output)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

}  // namespace

class InvertLabelsTask : public Task<InvertLabelsTask, INVERT_LABELS> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto labels  = context.input(0);
    auto classes = context.input(1);
    auto output  = context.output(0);

    legate::type_dispatch(legate::Type::Code::UINT64, invert_labels_fn{}, labels, classes, output);
  }

  static void gpu_variant(legate::TaskContext context)
  {
    auto labels  = context.input(0);
    auto classes = context.input(1);
    auto output  = context.output(0);

    legate::type_dispatch(
      labels.data().code(), invert_labels_fn_gpu{}, context, labels, classes, output);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::InvertLabelsTask::register_variants();
}

}  // namespace
