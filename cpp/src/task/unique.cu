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

#include <set>
#include <unordered_set>

#include "legate_library.h"
#include "legate_raft_cffi.h"

#include "legate/utilities/dispatch.h"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/matrix/copy.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace legate_raft {

namespace {

// TODO: Revert this back to unorderd_set after testing.
#define SET std::set

template <typename VAL>
void add_to_set(SET<VAL>& all_values, legate::PhysicalStore input)
{
  auto shape = input.shape<1>();
  auto acc   = input.read_accessor<VAL, 1>();
  for (legate::PointInRectIterator<1> it(shape, false /*fortran_order*/); it.valid(); ++it)
    all_values.insert(acc[*it]);
}

template <typename VAL>
void copy_to_output(legate::PhysicalStore output, const SET<VAL>& values)
{
  if (values.empty()) output.bind_empty_data();

  int64_t num_values = values.size();
  auto out_buf =
    output.create_output_buffer<VAL, 1>(legate::Point<1>(num_values), true /*bind_buffer*/);
  int64_t idx = 0;
  for (const auto& value : values)
    out_buf[idx++] = value;
}

template <legate::Type::Code CODE>
constexpr bool is_supported =
  !(legate::is_floating_point<CODE>::value || legate::is_complex<CODE>::value ||
    CODE == legate::Type::Code::FLOAT16);
struct unique_fn {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(legate::PhysicalStore output, std::vector<legate::PhysicalArray>& inputs)
  {
    using VAL = legate::type_of_t<CODE>;

    SET<VAL> all_values;
    // Find unique values across all inputs
    for (auto& input : inputs)
      add_to_set(all_values, input.data());
    // Copy the set of unique values to the output store
    copy_to_output(output, all_values);
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported<CODE>>* = nullptr>
  void operator()(legate::PhysicalStore output, std::vector<legate::PhysicalArray>& inputs)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

struct unique_fn_gpu {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(legate::PhysicalStore output,
                  std::vector<legate::PhysicalArray>& inputs,
                  cudaStream_t stream)
  {
    using VAL = legate::type_of_t<CODE>;

    raft::handle_t handle(stream);

    // Create temporary buffer for all input
    size_t size_all = 0;
    for (auto& input : inputs)
      size_all += input.shape<1>().volume();
    auto all_values = raft::make_device_vector<VAL>(handle, size_all);

    size_t size = 0;
    for (auto& input : inputs) {
      auto shape = input.data().shape<1>();
      auto acc   = input.data().read_accessor<VAL, 1>();

      // Consider to produce local set here.

      // Copy values from input to overall device vector
      auto values = raft::make_device_vector_view(acc.ptr(shape), shape.volume());

      raft::copy(all_values.data_handle() + size, values.data_handle(), shape.volume(), stream);
      size += shape.volume();
    }

    // // Produce set
    raft::device_span<VAL> all_values_span(all_values.data_handle(), all_values.size());
    thrust::sort(handle.get_thrust_policy(), all_values_span.begin(), all_values_span.end());
    auto end =
      thrust::unique(handle.get_thrust_policy(), all_values_span.begin(), all_values_span.end());

    // // Copy to output
    auto output_size = end - all_values_span.begin();
    auto out_buf     = output.create_output_buffer<VAL, 1>(legate::Point<1>(output_size), true);
    auto output_view = raft::make_device_vector_view<VAL, int32_t>(&out_buf[0], output_size);
    raft::copy(output_view.data_handle(), all_values.data_handle(), output_size, stream);
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported<CODE>>* = nullptr>
  void operator()(legate::PhysicalStore output,
                  std::vector<legate::PhysicalArray>& inputs,
                  cudaStream_t stream)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

}  // namespace

class UniqueTask : public Task<UniqueTask, UNIQUE> {
 public:
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context)
  {
    auto inputs = context.inputs();
    auto output = context.output(0);
    legate::type_dispatch(output.data().code(), unique_fn{}, output, inputs);
  }

  static void gpu_variant(legate::TaskContext context)
  {
    auto inputs = context.inputs();
    auto output = context.output(0);
    legate::type_dispatch(
      output.data().code(), unique_fn_gpu{}, output, inputs, context.get_task_stream());
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::UniqueTask::register_variants();
}

}  // namespace
