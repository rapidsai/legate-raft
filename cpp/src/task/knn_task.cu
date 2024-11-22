/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
 */

#include <cstdint>
#include <stdexcept>

#include "../legate_library.h"
#include "../legate_raft.h"

#include "common/gpu_task_context.hpp"
#include "common/utils.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/neighbors/brute_force.cuh>

#include <iostream>

namespace legate_raft {

namespace {

/**
 * @brief Reduce knn result from all nodes and fill the result on rank 0.
 *
 * This helper performs the tree-reduce over all nodes results via
 * NCCL communications.  Function currently assumes that multiple ranks exist.
 *
 * @param handle raft handle
 * @param n_rows The number of result rows in the buffer.
 * @param n_neighbors The number of neighbors (size of the buffers)
 * @param merge_indices_buf A buffer with the first half initialized to the
 *     knn indices result and the second half free to receive another result.
 * @param merge_distances_buf A buffer with the first half initialized to the
 *     knn distances result and the second half free to receive another result.
 * @param result_indices_view View for the final indices result,
 *     only written to for the first rank.
 * @param result_distances_view View for the final distances result,
 *     only written to for the first rank.
 */
template <typename DataT>
void perform_knn_tree_merge(const raft::handle_t& handle,
                            int64_t n_rows,
                            int64_t n_neighbors,
                            legate::Buffer<int64_t, 2>& merge_indices_buf,
                            legate::Buffer<DataT, 2>& merge_distances_buf,
                            raft::device_matrix_view<int64_t, int64_t>& result_indices_view,
                            raft::device_matrix_view<DataT, int64_t>& result_distances_view)
{
  const auto& comm = handle.get_comms();
  auto stream      = handle.get_stream();
  int n_ranks      = comm.get_size();
  int rank         = comm.get_rank();

  size_t part_size = n_rows * n_neighbors;
  size_t remaining = n_ranks;
  size_t radix     = 2;

  while (remaining > 1) {
    bool received_something = false;
    size_t offset           = radix / 2;

    comm.sync_stream(stream);

    comm.group_start();
    if (rank % radix == 0)  // This is one of the receivers
    {
      int other_id = rank + offset;
      if (other_id < n_ranks)  // Make sure someone's sending anything
      {
        // Receive others result into second half of merge buffer
        comm.device_recv(merge_indices_buf.ptr({n_rows, 0}), part_size, other_id, stream);
        comm.device_recv(merge_distances_buf.ptr({n_rows, 0}), part_size, other_id, stream);

        received_something = true;
      }
    } else if (rank % radix == offset)  // This is one of the senders
    {
      int other_id = rank - offset;
      comm.device_send(merge_indices_buf.ptr({0, 0}), part_size, other_id, stream);
      comm.device_send(merge_distances_buf.ptr({0, 0}), part_size, other_id, stream);
    }
    comm.group_end();
    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. This can result "
           "from a failed rank");

    remaining = (remaining + 1) / 2;
    radix *= 2;

    if (received_something) {
      // Update the local result by merging it with the remote one

      // We need the full local buffers for merging:
      auto merge_distances_view = raft::make_device_matrix_view<const DataT, int64_t>(
        merge_distances_buf.ptr({0, 0}), n_rows * 2, n_neighbors);
      auto merge_indices_view = raft::make_device_matrix_view<const int64_t, int64_t>(
        merge_indices_buf.ptr({0, 0}), n_rows * 2, n_neighbors);
      // And (again) their first half only to write the merge result:
      auto local_distances_view = raft::make_device_matrix_view<DataT, int64_t>(
        merge_distances_buf.ptr({0, 0}), n_rows, n_neighbors);
      auto local_indices_view = raft::make_device_matrix_view<int64_t, int64_t>(
        merge_indices_buf.ptr({0, 0}), n_rows, n_neighbors);

      // TODO: Remove translations. As of writing (raft 24.06) they are not
      //       actually optional.
      auto translations = std::vector<int64_t>({0, 0});
      auto trans        = raft::make_device_vector<int64_t, int64_t, raft::row_major>(handle, 2);
      raft::update_device(trans.data_handle(), translations.data(), 2, stream);
      auto trans_opt =
        std::make_optional<raft::device_vector_view<int64_t, int64_t>>(trans.data_handle(), 2);

      raft::neighbors::brute_force::knn_merge_parts(handle,
                                                    merge_distances_view,
                                                    merge_indices_view,
                                                    local_distances_view,
                                                    local_indices_view,
                                                    (size_t)n_rows,
                                                    trans_opt);

      // If done, copy the final result (uses local_*_view).
      if (remaining <= 1) {
        raft::copy(handle, result_indices_view, local_indices_view);
        raft::copy(handle, result_distances_view, local_distances_view);
      }
    }
  }
}

template <typename DataT>
static void knn_impl(raft::handle_t handle,
                     int64_t n_neighbors,
                     raft::distance::DistanceType distance_type,
                     legate::PhysicalStore index,
                     legate::PhysicalStore search,
                     legate::PhysicalStore indices_store,
                     legate::PhysicalStore distances_store)
{
  auto index_view  = device_matrix_view_from_store<const DataT, int64_t>(index, "index");
  auto index_vect  = std::vector({index_view});
  auto search_view = device_matrix_view_from_store<const DataT, int64_t>(search, "search");

  int64_t n_search_rows = search_view.extent(0);

  if (index_view.extent(0) < n_neighbors) {
    // NOTE: A mapper should ensure this (not that it happens in practice)
    throw std::runtime_error("individual node has too few data points");
  }

  int64_t result_size = 0;
  if (!handle.comms_initialized() || handle.get_comms().get_rank() == 0) {
    // Only rank 0 returns the result as of now (via late binding)
    // TODO: It would be nice to use a mapper to not need late-binding.
    result_size = n_search_rows;
  }

  // Create and bind the final results (empty/nullptr one if this isn't rank 0)
  auto res_indices_buf =
    indices_store.create_output_buffer<int64_t, 2>({result_size, n_neighbors}, true);
  auto res_distances_buf =
    distances_store.create_output_buffer<DataT, 2>({result_size, n_neighbors}, true);

  auto res_indices_view = raft::make_device_matrix_view<int64_t, int64_t>(
    result_size ? res_indices_buf.ptr({0, 0}) : nullptr, result_size, n_neighbors);
  auto res_distances_view = raft::make_device_matrix_view<DataT, int64_t>(
    result_size ? res_distances_buf.ptr({0, 0}) : nullptr, result_size, n_neighbors);

  /* If the task is running on a single GPU, call raft and return. */
  if (!handle.comms_initialized()) {
    raft::neighbors::brute_force::knn<int64_t, DataT, int64_t>(
      handle, index_vect, search_view, res_indices_view, res_distances_view, distance_type);
    return;
  } else if (handle.get_comms().get_size() == 1) {
    throw std::runtime_error("communication shouldn't be set up for singleton task.");
  }

  // To do the later tree-reduce, we need a buffer able to hold two nodes
  // results, write the local one in the first half and fetch the other half
  // from another node in the tree-reduce step:
  auto merge_indices_buf =
    indices_store.create_output_buffer<int64_t, 2>({n_search_rows * 2, n_neighbors}, false);
  auto merge_distances_buf =
    distances_store.create_output_buffer<DataT, 2>({n_search_rows * 2, n_neighbors}, false);

  // This run only sees part of the global index, so we need the offset
  int64_t index_offset = index.shape<2>().lo[0];

  // Write knn result into first half of the local indices and distances buf:
  auto local_indices_view = raft::make_device_matrix_view<int64_t, int64_t>(
    merge_indices_buf.ptr({0, 0}), n_search_rows, n_neighbors);
  auto local_distances_view = raft::make_device_matrix_view<DataT, int64_t>(
    merge_distances_buf.ptr({0, 0}), n_search_rows, n_neighbors);

  raft::neighbors::brute_force::knn<int64_t, DataT, int64_t>(handle,
                                                             index_vect,
                                                             search_view,
                                                             local_indices_view,
                                                             local_distances_view,
                                                             distance_type,
                                                             std::nullopt,
                                                             index_offset);

  perform_knn_tree_merge(handle,
                         n_search_rows,
                         n_neighbors,
                         merge_indices_buf,
                         merge_distances_buf,
                         res_indices_view,
                         res_distances_view);
}

}  // namespace

class RAFT_KNN_TASK : public Task<RAFT_KNN_TASK, RAFT_KNN> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    legate_raft::GPUTaskContext task_context{context};

    int64_t n_neighbors = context.scalar(0).value<int64_t>();
    std::string metric  = context.scalar(1).value<std::string>();

    auto index           = context.input(0).data();
    auto search          = context.input(1).data();
    auto indices_store   = context.output(0).data();
    auto distances_store = context.output(1).data();

    raft::distance::DistanceType distance_type;
    if (metric == "l2") {
      distance_type = raft::distance::DistanceType::L2SqrtExpanded;
    } else {
      throw std::invalid_argument("invalid or unimplemented metric");
    }

    if (!context.is_single_task()) {
      ncclComm_t nccl_com = *(context.communicators().at(0).get<ncclComm_t*>());
      task_context.inject_nccl_comm(nccl_com);
    }

    auto handle = task_context.handle();

    if (index.code() == legate::Type::Code::FLOAT32) {
      knn_impl<float>(
        handle, n_neighbors, distance_type, index, search, indices_store, distances_store);
    } else {
      knn_impl<double>(
        handle, n_neighbors, distance_type, index, search, indices_store, distances_store);
    }
  }
};

}  // namespace legate_raft

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  legate_raft::RAFT_KNN_TASK::register_variants();
}

}  // namespace
