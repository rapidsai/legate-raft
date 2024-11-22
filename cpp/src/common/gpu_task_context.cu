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

#include <common/gpu_task_context.hpp>

#include <raft/comms/std_comms.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/handle.hpp>

#include <nccl.h>

#include <cuda_runtime.h>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace legate_raft {

GPUTaskContext::GPUTaskContext(legate::TaskContext& context)
  : legate::dataframe::GPUTaskContext(context)
{
  handle_.reset(new raft::handle_t(stream()));
}

void GPUTaskContext::inject_nccl_comm(ncclComm_t nccl_comm)
{
  int n_ranks;
  RAFT_NCCL_TRY(ncclCommCount(nccl_comm, &n_ranks));

  int rank;
  RAFT_NCCL_TRY(ncclCommUserRank(nccl_comm, &rank));

  raft::comms::build_comms_nccl_only(handle_.get(), nccl_comm, n_ranks, rank);
}

raft::handle_t& GPUTaskContext::handle()
{
  if (!handle_) { handle_.reset(new raft::handle_t(stream())); }
  return *handle_;
}

}  // namespace legate_raft
