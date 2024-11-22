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

#include <stdexcept>

#include "legate_library.h"
#include "legate_raft_cffi.h"

#include <cudf/strings/strings_column_view.hpp>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/task_argument.hpp>

#include <common/gpu_task_context.hpp>
#include <thirdparty/murmur3.cuh>

#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#define N_ROWS_PER_BATCH     1048576
#define MAX_INSERTS_PER_ROWS 128
#define BLOCK_SIZE           256

namespace legate_raft {

using namespace legate::dataframe;

__device__ uint64_t compute_hash(const char* chars,
                                 const uint32_t start_offset,
                                 const uint32_t end_offset,
                                 const uint32_t seed)
{
  uint64_t hash_output[2];
  const char* input = &chars[start_offset];
  uint32_t length   = end_offset - start_offset;
  MurmurHash3_x64_128(input, length, seed, hash_output);
  return hash_output[0];
}

__device__ void insert(int32_t* output,
                       uint8_t* count,
                       uint32_t row,
                       uint32_t n_features,
                       uint32_t insert_count,
                       uint64_t hash)
{
  // no need for atomic operation as each row is handled by a single thread

  int32_t insertion  = hash % n_features;
  int32_t row_offset = row * MAX_INSERTS_PER_ROWS;

  // checking if insertion is already present
  for (int i = 0; i < insert_count; i++) {
    if (output[row_offset + i] == insertion) {
      count[row_offset + i] += 1;
      return;
    }
  }

  // first insertion
  output[row_offset + insert_count] = insertion;
  count[row_offset + insert_count]  = 1;
}

template <typename OFFSET_T>
__global__ void hashing_kernel(const OFFSET_T* offsets,
                               const char* chars,
                               uint32_t row_offset,
                               int32_t* output,
                               uint8_t* count,
                               uint32_t n_rows,
                               uint32_t n_features,
                               const uint32_t seed)
{
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_rows) return;

  uint32_t insert_count = 0;
  uint32_t start_offset = offsets[row_offset + row];
  uint32_t end_offset   = offsets[row_offset + row + 1];
  for (int i = start_offset; i < end_offset; i++) {
    if (chars[i] == ' ') {
      if (i - start_offset > 1) {  // not empty word (double space)
        uint64_t hash = compute_hash(chars, start_offset, i, seed);
        insert(output, count, row, n_features, insert_count++, hash);
      }
      start_offset = i + 1;
    }
  }
  if (end_offset - start_offset > 1) {  // not empty word (double space)
    uint64_t hash = compute_hash(chars, start_offset, end_offset, seed);
    insert(output, count, row, n_features, insert_count++, hash);
  }
}

__global__ void sparsify(int32_t* output,
                         uint8_t* count,
                         float* out_data,
                         uint64_t* out_row,
                         uint64_t* out_col,
                         uint64_t* nnz,
                         uint64_t n_elms,
                         uint64_t row_offset)
{
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_elms) return;

  int32_t insertion = output[idx];
  if (insertion >= 0) {
    uint64_t old_nnz = atomicAdd((unsigned long long int*)nnz, (unsigned long long int)1);
    uint64_t row     = idx / MAX_INSERTS_PER_ROWS;

    out_data[old_nnz] = count[idx];
    out_row[old_nnz]  = row_offset + row;
    out_col[old_nnz]  = insertion;
  }
}

class HashingVectorizerTask : public Task<HashingVectorizerTask, HASHING_VECTORIZER> {
 public:
  static void gpu_variant(legate::TaskContext ctx)
  {
    legate_raft::GPUTaskContext gpu_task_context{ctx};
    const auto input = argument::get_next_input<task::PhysicalColumn>(gpu_task_context);
    cudf::strings_column_view col{input.column_view()};

    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    // Fetch input offsets (may be a 32bit or 64bit integer column)
    auto offsets = col.offsets();

    // outputs
    auto out_data = ctx.output(0).data();
    auto out_row  = ctx.output(1).data();
    auto out_col  = ctx.output(2).data();

    // scalars
    auto [scalar_off, input_off, output_off] = gpu_task_context.get_task_argument_indices();
    int n_features                           = ctx.scalar(scalar_off++).value<int>();
    int seed                                 = ctx.scalar(scalar_off++).value<int>();

    uint64_t row_offset        = input.global_row_offset();
    uint64_t n_rows_to_process = col.size();

    rmm::device_scalar<uint64_t> d_nnz(stream);
    uint64_t h_nnz = 0;
    raft::copy(d_nnz.data(), &h_nnz, 1, stream);

    auto out_data_vec =
      new rmm::device_uvector<float>(n_rows_to_process * MAX_INSERTS_PER_ROWS, stream);
    auto out_row_vec =
      new rmm::device_uvector<uint64_t>(n_rows_to_process * MAX_INSERTS_PER_ROWS, stream);
    auto out_col_vec =
      new rmm::device_uvector<uint64_t>(n_rows_to_process * MAX_INSERTS_PER_ROWS, stream);

    RAFT_CUDA_TRY(cudaMemsetAsync(
      out_data_vec->data(), 0, n_rows_to_process * MAX_INSERTS_PER_ROWS * sizeof(float), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      out_row_vec->data(), 0, n_rows_to_process * MAX_INSERTS_PER_ROWS * sizeof(uint64_t), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      out_col_vec->data(), 0, n_rows_to_process * MAX_INSERTS_PER_ROWS * sizeof(uint64_t), stream));

    uint32_t n_batches = (n_rows_to_process + N_ROWS_PER_BATCH - 1) / N_ROWS_PER_BATCH;
    for (uint32_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
      uint64_t batch_offset = batch_idx * N_ROWS_PER_BATCH;
      uint64_t n_rows_of_batch =
        std::min((uint64_t)N_ROWS_PER_BATCH, n_rows_to_process - batch_offset);
      uint64_t n_batch_elms = n_rows_of_batch * MAX_INSERTS_PER_ROWS;

      auto output_buf = rmm::device_uvector<int32_t>(n_batch_elms, stream);
      RAFT_CUDA_TRY(cudaMemsetAsync(output_buf.data(), -1, n_batch_elms * sizeof(int32_t), stream));

      auto count_buf = rmm::device_uvector<uint8_t>(n_batch_elms, stream);
      RAFT_CUDA_TRY(cudaMemsetAsync(count_buf.data(), 0, n_batch_elms * sizeof(uint8_t), stream));

      int num_blocks = (n_rows_of_batch + BLOCK_SIZE - 1) / BLOCK_SIZE;
      if (offsets.type().id() == cudf::type_id::INT32) {
        hashing_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(offsets.data<int32_t>(),
                                                              col.chars_begin(stream),
                                                              batch_offset,
                                                              output_buf.data(),
                                                              count_buf.data(),
                                                              n_rows_of_batch,
                                                              n_features,
                                                              seed);
      } else if (offsets.type().id() == cudf::type_id::INT64) {
        hashing_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(offsets.data<int64_t>(),
                                                              col.chars_begin(stream),
                                                              batch_offset,
                                                              output_buf.data(),
                                                              count_buf.data(),
                                                              n_rows_of_batch,
                                                              n_features,
                                                              seed);
      } else {
        throw std::runtime_error("internal error: invalid offset type.");
      }

      num_blocks = (n_batch_elms + BLOCK_SIZE - 1) / BLOCK_SIZE;
      sparsify<<<num_blocks, BLOCK_SIZE, 0, stream>>>(output_buf.data(),
                                                      count_buf.data(),
                                                      out_data_vec->data(),
                                                      out_row_vec->data(),
                                                      out_col_vec->data(),
                                                      d_nnz.data(),
                                                      n_batch_elms,
                                                      row_offset + batch_offset);

      handle.sync_stream();
    }

    raft::copy(&h_nnz, d_nnz.data(), 1, stream);
    handle.sync_stream();

    auto out_data_buf = out_data.create_output_buffer<float, 1>(legate::Point<1>(h_nnz), true);
    raft::copy(&out_data_buf[0], out_data_vec->data(), h_nnz, stream);
    handle.sync_stream();
    delete out_data_vec;

    auto out_row_buf = out_row.create_output_buffer<uint64_t, 1>(legate::Point<1>(h_nnz), true);
    raft::copy(&out_row_buf[0], out_row_vec->data(), h_nnz, stream);
    handle.sync_stream();
    delete out_row_vec;

    auto out_col_buf = out_col.create_output_buffer<uint64_t, 1>(legate::Point<1>(h_nnz), true);
    raft::copy(&out_col_buf[0], out_col_vec->data(), h_nnz, stream);
    handle.sync_stream();
    delete out_col_vec;

    handle.sync_stream();
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::HashingVectorizerTask::register_variants();
}

}  // namespace
