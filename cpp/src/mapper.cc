/* Copyright 2025 NVIDIA Corporation
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

#include "mapper.h"
#include "legate_raft.h"
#include "legate_raft_cffi.h"
#include <legate.h>
#include <set>
#include <vector>

namespace legate_raft {

LegateRaftMapper::LegateRaftMapper() = default;

auto LegateRaftMapper::allocation_pool_size(const legate::mapping::Task& task,
                                            legate::mapping::StoreTarget memory_kind)
  -> std::optional<std::size_t>
{
  const auto task_id = static_cast<int>(task.task_id());

  if (memory_kind == legate::mapping::StoreTarget::ZCMEM) {
    switch (task_id) {
      case RAFT_KMEANS_FIT:
        // This task uses a small allocation for result metadata.
        return 2 * 8;
      default: return 0;
    }
  }
  // TODO(seberg): nullopt means we give no upper bound.  For tasks that use
  // `legate::VariantOptions{}.with_has_allocations(true);` giving a bound
  // may improve parallelism.
  return std::nullopt;
}

auto LegateRaftMapper::tunable_value(legate::TunableID /*tunable_id*/) -> legate::Scalar
{
  return legate::Scalar{};
}

auto LegateRaftMapper::store_mappings(const legate::mapping::Task& task,
                                      const std::vector<legate::mapping::StoreTarget>& options)
  -> std::vector<legate::mapping::StoreMapping>
{
  auto task_id = task.task_id();
  // Enforce c-ordering for these tasks on the task level
  const std::set<LegateRaftOpCode> row_major_only({RAFT_KMEANS_FIT, RAFT_KMEANS_PREDICT, RAFT_KNN});
  std::vector<legate::mapping::StoreMapping> mappings;
  if (row_major_only.count(static_cast<LegateRaftOpCode>(task_id)) != 0U) {
    for (auto input : task.inputs()) {
      mappings.push_back(
        legate::mapping::StoreMapping::default_mapping(input.data(), options.front()));
      mappings.back().policy().ordering.set_c_order();
      mappings.back().policy().exact = true;
    }
    return mappings;
  }
  return mappings;
}

}  // namespace legate_raft