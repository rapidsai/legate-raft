/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include "../legate_library.h"
#include "../legate_raft.h"

#include "kmeans_mnmg_impl.cuh"
#include "raft_kmeans_api.hpp"

#include <nccl.h>

namespace kmeans {
// ----------------------------- fit ---------------------------------//

template <typename DataT, typename IndexT>
void fit(raft::handle_t const& handle,
         const raft::cluster::KMeansParams& params,
         raft::device_matrix_view<const DataT, IndexT> X,
         std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter)
{
  const DataT* weights = nullptr;
  if (sample_weight.has_value()) { weights = sample_weight.value().data_handle(); }

  handle.get_comms().barrier();
  // TODO (seberg): Should push the device matrix/vector views all the way through here:
  impl::fit(handle,
            params,
            X.data_handle(),
            X.extent(0),
            X.extent(1),
            weights,
            centroids.data_handle(),
            inertia,
            n_iter);

  handle.sync_stream();
}

template void fit(raft::handle_t const& handle,
                  const raft::cluster::kmeans::KMeansParams&,
                  raft::device_matrix_view<const float, int> X,
                  std::optional<raft::device_vector_view<const float, int>>,
                  raft::device_matrix_view<float, int>,
                  raft::host_scalar_view<float>,
                  raft::host_scalar_view<int>);

template void fit(raft::handle_t const& handle,
                  const raft::cluster::kmeans::KMeansParams&,
                  raft::device_matrix_view<const double, int>,
                  std::optional<raft::device_vector_view<const double, int>>,
                  raft::device_matrix_view<double, int>,
                  raft::host_scalar_view<double>,
                  raft::host_scalar_view<int>);

};  // end namespace kmeans
