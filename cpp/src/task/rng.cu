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
#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

namespace legate_raft {

namespace {

template <legate::Type::Code CODE>
constexpr bool is_supported =
  (CODE == legate::Type::Code::INT32 || CODE == legate::Type::Code::INT64 ||
   CODE == legate::Type::Code::UINT32 || CODE == legate::Type::Code::UINT64);

struct uniform_int_fn_gpu {
  template <legate::Type::Code CODE, int32_t DIM, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::Scalar random_seed,
                  legate::Scalar start,
                  legate::Scalar end)
  {
    using VAL = legate::type_of_t<CODE>;

    auto shape = output.shape<DIM>();
    if (shape.empty()) return;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    auto output_acc  = output.write_accessor<VAL, DIM>();
    auto output_view = raft::make_device_vector_view(output_acc.ptr(shape), shape.volume());

    raft::random::RngState rng_state{random_seed.value<uint64_t>() + shape.lo[0]};
    raft::random::uniformInt(handle, rng_state, output_view, start.value<VAL>(), end.value<VAL>());
  }

  template <legate::Type::Code CODE, int32_t DIM, std::enable_if_t<!is_supported<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::Scalar random_seed,
                  legate::Scalar start,
                  legate::Scalar end)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

}  // namespace

class UniformIntTask : public Task<UniformIntTask, UNIFORM_INT> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    auto random_seed = context.scalar(0);
    auto start       = context.scalar(1);
    auto end         = context.scalar(2);

    auto output = context.output(0);

    legate::double_dispatch(output.data().dim(),
                            output.data().code(),
                            uniform_int_fn_gpu{},
                            context,
                            output,
                            random_seed,
                            start,
                            end);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::UniformIntTask::register_variants();
}

}  // namespace
