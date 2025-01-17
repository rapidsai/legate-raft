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

#include "legate/type/type_traits.h"
#include "legate/utilities/dispatch.h"

namespace legate_raft {

namespace {

struct categorize_fn {
  template <legate::Type::Code CODE, std::enable_if_t<!legate::is_complex<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore result,
                  legate::PhysicalStore input,
                  legate::PhysicalStore bins)
  {
    using VAL = legate::type_of_t<CODE>;

    auto in_shape  = result.shape<1>();
    auto bin_shape = bins.shape<1>();

    assert(!bin_shape.empty());
    if (in_shape.empty()) return;

    auto num_bins = bin_shape.hi[0] - bin_shape.lo[0];

    auto in_acc  = input.read_accessor<VAL, 1>();
    auto bin_acc = bins.read_accessor<VAL, 1>();
    auto res_acc = result.write_accessor<uint64_t, 1>();

    for (legate::PointInRectIterator<1> it(in_shape); it.valid(); ++it) {
      auto p      = *it;
      auto& value = in_acc[p];
      for (auto bin_idx = 0; bin_idx < num_bins; ++bin_idx) {
        if (bin_acc[bin_idx] <= value && value < bin_acc[bin_idx + 1]) {
          res_acc[p] = static_cast<uint64_t>(bin_idx);
          break;
        }
      }
    }
  }

  template <legate::Type::Code CODE, std::enable_if_t<legate::is_complex<CODE>::value>* = nullptr>
  void operator()(legate::PhysicalStore result,
                  legate::PhysicalStore input,
                  legate::PhysicalStore bins)
  {
    assert(false);
  }
};

}  // namespace

class CategorizeTask : public Task<CategorizeTask, CATEGORIZE> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto bins   = context.input(1);
    auto result = context.output(0);

    legate::type_dispatch(input.data().code(), categorize_fn{}, result, input, bins);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::CategorizeTask::register_variants();
}

}  // namespace
