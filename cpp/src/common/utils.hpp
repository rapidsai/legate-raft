/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <stdexcept>
#include <type_traits>

#include <legate.h>
#include <raft/core/device_mdspan.hpp>

namespace legate_raft {

namespace {

/*
 * Helper to either get a read or a read-write accessor depending on whether
 * the DataT is const or not.
 */
template <typename DataT, int dims>
struct get_accessor {
  template <typename D = DataT, std::enable_if_t<std::is_const<D>::value, bool> = true>
  auto operator()(legate::PhysicalStore& store)
  {
    return store.read_accessor<D, dims, true>();
  }

  template <typename D = DataT, std::enable_if_t<!std::is_const<D>::value, bool> = true>
  auto operator()(legate::PhysicalStore& store)
  {
    return store.read_write_accessor<D, dims, true>();
  }
};

}  // namespace

/**
 * @brief create a raft matrix view from a physical array.  The full store
 * is viewed and it must be 2 dimensional.
 *
 * A read-only accessor is used if DataT is const qualified, a read-write
 * accessor if it isn't.
 * Currently, assumes and checks for row-major (C-order).
 *
 * @param arr The legate physical array.
 * @param name A name to use for an error if the store isn't row-major.
 * @return matrix_view
 */
template <typename DataT, typename IndexT>
auto constexpr device_matrix_view_from_store(legate::PhysicalStore& store,
                                             const char* name = "data")
{
  const auto shape = store.shape<2>();

  auto rows = shape.hi[0] - shape.lo[0] + 1;
  auto cols = shape.hi[1] - shape.lo[1] + 1;
  if ((IndexT)rows != rows || (IndexT)cols != cols) {
    throw std::runtime_error(name + std::string(" size too large for local matrix view."));
  }

  auto accessor = get_accessor<DataT, 2>{}(store);
  if (!accessor.accessor.is_dense_row_major(shape)) {
    throw std::invalid_argument(name +
                                std::string(" must be stored row-major, please make a copy."));
  }

  return raft::make_device_matrix_view<DataT, IndexT>(
    accessor.ptr(shape), (IndexT)rows, (IndexT)cols);
}

/**
 * @brief create a raft vector view from a physical store.  The full store
 * is viewed and it must be 1 dimensional.
 *
 * A read-only accessor is used if DataT is const qualified, a read-write
 * accessor if it isn't.
 * If the store has more than 1 dimensions, use the `store_dims` template
 * parameter to ignore all but the leading dimension.
 *
 * TODO: The dimension reductin mechanism isn't great, but I couldn't think
 *       of a better way to do this thing (maybe we can apss in 1-D also?)
 *
 * @param arr The legate physical array.
 * @param name A name to use for an error if the store isn't row-major.
 * @return vector_view
 */
template <typename DataT, typename IndexT, int store_dims = 1>
auto constexpr device_vector_view_from_store(legate::PhysicalStore& store,
                                             const char* name = "data")
{
  auto shape = store.shape<store_dims>();

  for (int i = 1; i < store_dims; i++) {
    shape.hi[i] = shape.lo[i];
  }

  auto length = shape.hi[0] - shape.lo[0] + 1;
  if ((IndexT)length != length) {
    throw std::runtime_error(name + std::string(" size too large for local vector view."));
  }

  auto accessor = get_accessor<DataT, store_dims>{}(store);
  if (!accessor.accessor.is_dense_row_major(shape)) {
    throw std::invalid_argument(name +
                                std::string(" must be stored row-major, please make a copy."));
  }

  return raft::make_device_vector_view<DataT, IndexT>(accessor.ptr(shape), (IndexT)length);
}

}  // namespace legate_raft
