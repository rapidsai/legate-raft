/* Copyright 2021 NVIDIA Corporation
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

#include "../legate_library.h"
#include "../legate_raft.h"

#include <raft/core/handle.hpp>

#include <nccl.h>

#include <legate_dataframe/core/task_context.hpp>

#include <cuda_runtime.h>

namespace legate_raft {

// This helper class is to make sure that each GPU task uses its own allocator
// for temporary allocations from libraft during its execution. This class also
// creates a fresh stream to be used for kernels.

class GPUTaskContext : public legate::dataframe::GPUTaskContext {
 public:
  GPUTaskContext(legate::TaskContext& context);
  void inject_nccl_comm(ncclComm_t nccl_comm);
  raft::handle_t& handle();

 private:
  std::unique_ptr<raft::handle_t> handle_{nullptr};
};

}  // namespace legate_raft
