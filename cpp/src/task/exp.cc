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

#include "legate/utilities/dispatch.h"

namespace legate_raft {

namespace {

struct exp_fn {
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
      output_acc[p] = exp(input_acc[p]);
    }
  }
};

}  // namespace

class ExpTask : public Task<ExpTask, EXP> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto output = context.output(0);

    legate::double_dispatch(input.data().dim(), input.data().code(), exp_fn{}, output, input);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::ExpTask::register_variants();
}

}  // namespace
