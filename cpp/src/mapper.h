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
#pragma once

#include "legate_library.h"
#include <vector>

namespace legate_raft {

class LegateRaftMapper : public legate::mapping::Mapper {
 public:
  LegateRaftMapper();
  ~LegateRaftMapper() override                                     = default;
  LegateRaftMapper(const LegateRaftMapper& rhs)                    = delete;
  LegateRaftMapper(LegateRaftMapper&& rhs)                         = delete;
  auto operator=(const LegateRaftMapper& rhs) -> LegateRaftMapper& = delete;
  auto operator=(LegateRaftMapper&& rhs) -> LegateRaftMapper&      = delete;

  // Legate mapping functions

  auto allocation_pool_size(const legate::mapping::Task& task,
                            legate::mapping::StoreTarget memory_kind)
    -> std::optional<std::size_t> override;
  auto store_mappings(const legate::mapping::Task& task,
                      const std::vector<legate::mapping::StoreTarget>& options)
    -> std::vector<legate::mapping::StoreMapping> override;
  auto tunable_value(legate::TunableID tunable_id) -> legate::Scalar override;
};

}  // namespace legate_raft
