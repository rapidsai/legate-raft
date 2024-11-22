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
#include <raft/matrix/argmax.cuh>

namespace legate_raft {

namespace {

struct argmax_reduction_fn {
  template <legate::Type::Code CODE>
  void operator()(legate::PhysicalStore input, legate::PhysicalStore output)
  {
    using VAL = legate::type_of_t<CODE>;

    const auto input_shape = input.shape<2>();
    if (input_shape.empty()) return;

    auto in_acc  = input.read_accessor<VAL, 2>();
    auto out_acc = output.write_accessor<int64_t, 1>();

    for (legate::PointInRectIterator<2> it(input_shape, false); it.valid(); ++it) {
      auto value = in_acc[*it];

      auto row = it[0];
      auto col = it[1];
      legate::Point<1> p(row);
      legate::Point<2> p_max(row, out_acc[p]);

      if (value > in_acc[p_max]) { out_acc[p] = col; }
    }
  }
};

template <legate::Type::Code CODE>
constexpr bool is_supported_gpu =
  (CODE == legate::Type::Code::FLOAT32 || CODE == legate::Type::Code::FLOAT64);

struct argmax_reduction_fn_gpu {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore input,
                  legate::PhysicalStore output)
  {
    using VAL = legate::type_of_t<CODE>;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    auto in_acc  = input.read_accessor<VAL, 2>();
    auto out_acc = output.write_accessor<int64_t, 1>();

    auto input_shape  = input.shape<2>();
    auto output_shape = output.shape<1>();
    int64_t n_rows    = input_shape.hi[0] - input_shape.lo[0] + 1;
    int64_t n_cols    = input_shape.hi[1] - input_shape.lo[1] + 1;

    if (n_rows > 0) {
      auto input_view = raft::make_device_matrix_view<const VAL, int64_t, raft::row_major>(
        in_acc.ptr(input_shape), n_rows, n_cols);
      auto output_view =
        raft::make_device_vector_view<int64_t, int64_t>(out_acc.ptr(output_shape), n_rows);

      raft::matrix::argmax(handle, input_view, output_view);
      handle.sync_stream();
    }
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore input,
                  legate::PhysicalStore output)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

}  // namespace

class ArgMaxTask : public Task<ArgMaxTask, ARG_MAX> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto result = context.output(0);
    // if (result.data().transformed()) { result.data().remove_transform(); }

    legate::type_dispatch(input.data().code(), argmax_reduction_fn{}, input.data(), result.data());
  }

  static void gpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto result = context.output(0);
    // if (result.transformed()) { result.remove_transform(); }

    legate::type_dispatch(
      input.data().code(), argmax_reduction_fn_gpu{}, context, input.data(), result.data());
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::ArgMaxTask::register_variants();
}

}  // namespace
