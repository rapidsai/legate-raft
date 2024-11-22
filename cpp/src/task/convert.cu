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

#include <common/gpu_task_context.hpp>

#include "legate/utilities/dispatch.h"
#include "legate/utilities/typedefs.h"

namespace legate_raft {

namespace {

template <legate::Type::Code SRC_TYPE, legate::Type::Code DST_TYPE>
struct convert_fn_cpu {
  template <int32_t DIM>
  void operator()(legate::PhysicalStore output, legate::PhysicalStore input)
  {
    using SRC = legate::type_of_t<SRC_TYPE>;
    using DST = legate::type_of_t<DST_TYPE>;

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto input_acc  = input.read_accessor<SRC, DIM>();
    auto output_acc = output.write_accessor<DST, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran order*/); it.valid(); ++it) {
      auto p        = *it;
      output_acc[p] = static_cast<DST>(input_acc[p]);
    }
  }
};

template <typename value_in_t, typename value_out_t>
__global__ void convert_kernel(const value_in_t* in, value_out_t* out, uint64_t volume)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < volume) out[idx] = static_cast<value_out_t>(in[idx]);
}

template <legate::Type::Code SRC_TYPE, legate::Type::Code DST_TYPE>
struct convert_fn_gpu {
  template <int32_t DIM>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::PhysicalStore input)
  {
    using SRC = legate::type_of_t<SRC_TYPE>;
    using DST = legate::type_of_t<DST_TYPE>;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto input_acc  = input.read_accessor<SRC, DIM>();
    auto output_acc = output.write_accessor<DST, DIM>();

    auto volume = shape.volume();

    int block_size = 256;  // TODO: tune
    int num_blocks = (volume + block_size - 1) / block_size;

    convert_kernel<<<num_blocks, block_size, 0, stream>>>(
      input_acc.ptr(shape), output_acc.ptr(shape), volume);
    handle.sync_stream();
  }
};

}  // namespace

class ConvertTask : public Task<ConvertTask, CONVERT> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto output = context.output(0);

    switch (input.data().code()) {
      case legate::Type::Code::UINT64:
        switch (output.data().code()) {
          case legate::Type::Code::FLOAT64:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_cpu<legate::Type::Code::UINT64, legate::Type::Code::FLOAT64>{},
              output,
              input);
          default: throw(std::runtime_error("Cannot convert to that output type."));
        }
      case legate::Type::Code::INT64:
        switch (output.data().code()) {
          case legate::Type::Code::INT32:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_cpu<legate::Type::Code::INT64, legate::Type::Code::INT32>{},
              output,
              input);
          case legate::Type::Code::FLOAT32:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_cpu<legate::Type::Code::INT64, legate::Type::Code::FLOAT32>{},
              output,
              input);
          case legate::Type::Code::FLOAT64:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_cpu<legate::Type::Code::INT64, legate::Type::Code::FLOAT64>{},
              output,
              input);
          default: throw(std::runtime_error("Cannot convert to that output type."));
        }
      case legate::Type::Code::INT32:
        switch (output.data().code()) {
          case legate::Type::Code::UINT32:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_cpu<legate::Type::Code::INT32, legate::Type::Code::UINT32>{},
              output,
              input);
          case legate::Type::Code::UINT64:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_cpu<legate::Type::Code::INT32, legate::Type::Code::UINT64>{},
              output,
              input);
          case legate::Type::Code::FLOAT32:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_cpu<legate::Type::Code::INT32, legate::Type::Code::FLOAT32>{},
              output,
              input);
          case legate::Type::Code::FLOAT64:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_cpu<legate::Type::Code::INT32, legate::Type::Code::FLOAT64>{},
              output,
              input);
          default: throw(std::runtime_error("Cannot convert to that output type."));
        }
      case legate::Type::Code::FLOAT64:
        switch (output.data().code()) {
          case legate::Type::Code::FLOAT32:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_cpu<legate::Type::Code::FLOAT64, legate::Type::Code::FLOAT32>{},
              output,
              input);
          default: throw(std::runtime_error("Cannot convert to that output type."));
        }
      default: throw(std::runtime_error("Cannot convert from this input type."));
    }
  }

  static void gpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto output = context.output(0);

    switch (input.data().code()) {
      case legate::Type::Code::INT32:
        switch (output.data().code()) {
          case legate::Type::Code::INT64:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::INT32, legate::Type::Code::INT64>{},
              context,
              output,
              input);
          case legate::Type::Code::UINT32:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::INT32, legate::Type::Code::UINT32>{},
              context,
              output,
              input);
          case legate::Type::Code::UINT64:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::INT32, legate::Type::Code::UINT64>{},
              context,
              output,
              input);
          case legate::Type::Code::FLOAT32:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::INT32, legate::Type::Code::FLOAT32>{},
              context,
              output,
              input);
          case legate::Type::Code::FLOAT64:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::INT32, legate::Type::Code::FLOAT64>{},
              context,
              output,
              input);
          default: throw(std::runtime_error("Cannot convert to that output type."));
        }
      case legate::Type::Code::UINT64:
        switch (output.data().code()) {
          case legate::Type::Code::INT32:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::UINT64, legate::Type::Code::INT32>{},
              context,
              output,
              input);
          case legate::Type::Code::INT64:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::UINT64, legate::Type::Code::INT64>{},
              context,
              output,
              input);
          case legate::Type::Code::FLOAT32:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::UINT64, legate::Type::Code::FLOAT32>{},
              context,
              output,
              input);
          case legate::Type::Code::FLOAT64:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::UINT64, legate::Type::Code::FLOAT64>{},
              context,
              output,
              input);
          default: throw(std::runtime_error("Cannot convert to that output type."));
        }
      case legate::Type::Code::INT64:
        switch (output.data().code()) {
          case legate::Type::Code::INT32:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::INT64, legate::Type::Code::INT32>{},
              context,
              output,
              input);
          case legate::Type::Code::FLOAT32:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::INT64, legate::Type::Code::FLOAT32>{},
              context,
              output,
              input);
          case legate::Type::Code::FLOAT64:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::INT64, legate::Type::Code::FLOAT64>{},
              context,
              output,
              input);
          default: throw(std::runtime_error("Cannot convert to that output type."));
        }
      case legate::Type::Code::FLOAT32:
        switch (output.data().code()) {
          case legate::Type::Code::FLOAT64:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::FLOAT32, legate::Type::Code::FLOAT64>{},
              context,
              output,
              input);
          default: throw(std::runtime_error("Cannot convert to that output type."));
        }
      case legate::Type::Code::FLOAT64:
        switch (output.data().code()) {
          case legate::Type::Code::FLOAT32:
            return legate::dim_dispatch(
              input.data().dim(),
              convert_fn_gpu<legate::Type::Code::FLOAT64, legate::Type::Code::FLOAT32>{},
              context,
              output,
              input);
          default: throw(std::runtime_error("Cannot convert to that output type."));
        }
      default: throw(std::runtime_error("Cannot convert from this input type."));
    }
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::ConvertTask::register_variants();
}

}  // namespace
