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
#include <raft/core/device_mdarray.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

namespace legate_raft {

namespace {

template <legate::Type::Code CODE>
constexpr bool is_supported =
  (legate::is_integral<CODE>::value && CODE != legate::Type::Code::BOOL);

struct map_labels_fn {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(legate::PhysicalStore labels,
                  legate::PhysicalStore classes,
                  legate::PhysicalStore output)
  {
    using VAL = legate::type_of_t<CODE>;

    const auto shape         = labels.shape<1>();
    const auto classes_shape = classes.shape<1>();
    const auto n_classes     = classes_shape.hi + 1;

    auto labels_acc  = labels.read_accessor<VAL, 1>();
    auto classes_acc = classes.read_accessor<VAL, 1>();
    auto output_acc  = output.write_accessor<VAL, 1>();

    bool found = false;

    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      found = false;
      for (VAL idx = 0; idx < n_classes; ++idx) {
        auto label  = labels_acc[*it];
        auto class_ = classes_acc[idx];
        if (label == class_) {
          output_acc[*it] = idx;
          found           = true;
          break;
        }
      }
      // Failed to find label among provided classes.
      if (!found) throw std::runtime_error("Mismatch between provided labels and classes.");
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
__global__ void map_labels_kernel(wo_t output_acc,
                                  ro_t labels_acc,
                                  ro_t classes_acc,
                                  size_t volume,
                                  size_t offset,
                                  size_t n_classes,
                                  bool* failure_to_match)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= volume) return;
  auto idx   = tid + offset;
  auto label = labels_acc[idx];
  for (int j = 0; j < n_classes; j++) {
    auto class_ = classes_acc[j];
    if (label == class_) {
      output_acc[idx] = j;
      return;
    }
  }
  *failure_to_match = true;  // error condition
}

struct map_labels_fn_gpu {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore labels,
                  legate::PhysicalStore classes,
                  legate::PhysicalStore output)
  {
    using VAL = legate::type_of_t<CODE>;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    const auto labels_shape = labels.shape<1>();
    const auto volume       = labels_shape.volume();
    const auto offset       = labels_shape.lo[0];

    const auto classes_shape = classes.shape<1>();
    // const auto n_classes = classes_shape.hi + 1;
    const auto n_classes = classes_shape.volume();
    assert(classes_shape.lo[0] == 0);  // assume classes are broadcast

    auto labels_acc  = labels.read_accessor<VAL, 1>();
    auto classes_acc = classes.read_accessor<VAL, 1>();
    auto output_acc  = output.write_accessor<VAL, 1>();

    auto failure_to_match_device = raft::make_device_scalar<bool>(handle, false);
    bool failure_to_match        = false;

    int block_size = 256;  // TODO: tune
    int num_blocks = (volume + block_size - 1) / block_size;
    map_labels_kernel<<<num_blocks, block_size, 0, stream>>>(output_acc,
                                                             labels_acc,
                                                             classes_acc,
                                                             volume,
                                                             offset,
                                                             n_classes,
                                                             failure_to_match_device.data_handle());

    RAFT_CUDA_TRY(cudaMemcpyAsync(&failure_to_match,
                                  failure_to_match_device.data_handle(),
                                  sizeof(bool),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    if (failure_to_match) throw std::runtime_error("Mismatch between provided labels and classes.");
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

class MapLabelsTask : public Task<MapLabelsTask, MAP_LABELS> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto labels  = context.input(0);
    auto classes = context.input(1);
    auto output  = context.output(0);

    legate::type_dispatch(labels.data().code(), map_labels_fn{}, labels, classes, output);
  }

  static void gpu_variant(legate::TaskContext context)
  {
    auto labels  = context.input(0);
    auto classes = context.input(1);
    auto output  = context.output(0);

    legate::type_dispatch(
      labels.data().code(), map_labels_fn_gpu{}, context, labels, classes, output);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::MapLabelsTask::register_variants();
}

}  // namespace
