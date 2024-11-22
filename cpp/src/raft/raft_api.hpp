/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#pragma once

#include <string>

template <typename idx_t, typename value_t>
void raft_knn(idx_t n_index_rows,
              idx_t n_search_rows,
              idx_t n_features,
              idx_t k,
              std::string& metric,
              const value_t* index_ptr,
              const value_t* search_ptr,
              idx_t* indices_ptr,
              float* distances_ptr);

template <typename idx_t>
void raft_knn_merge(size_t n_samples,
                    int n_parts,
                    int k,
                    const idx_t* in_ind,
                    const float* in_dist,
                    idx_t* out_ind,
                    float* out_dist);
