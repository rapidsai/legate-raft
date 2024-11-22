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
#include <cmath>

#include "legate_library.h"
#include "legate_raft_cffi.h"

#include <common/gpu_task_context.hpp>

#include "legate/utilities/dispatch.h"

namespace legate_raft {

namespace {

template <legate::Type::Code CODE>
constexpr bool is_supported_gpu =
  (CODE == legate::Type::Code::FLOAT32 || CODE == legate::Type::Code::FLOAT64);

template <typename output_acc_t, typename input_acc_t>
__global__ void tfidf_fit_kernel(
  output_acc_t out, input_acc_t in, int32_t smooth_idf, uint64_t n_samples, size_t volume)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;

  out[idx] = 1.0 - log((in[idx] + smooth_idf) / (n_samples + smooth_idf));
}

struct tfidf_fit_fn_gpu {
  template <legate::Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::PhysicalStore input,
                  legate::Scalar smooth_idf,
                  legate::Scalar n_samples)
  {
    using VAL = legate::type_of_t<CODE>;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto volume = shape.volume();

    int block_size = 256;  // TODO: tune
    int num_blocks = (volume + block_size - 1) / block_size;

    auto input_acc  = input.read_accessor<VAL, DIM>();
    auto output_acc = output.write_accessor<VAL, DIM>();

    tfidf_fit_kernel<<<num_blocks, block_size, 0, stream>>>(output_acc.ptr(shape),
                                                            input_acc.ptr(shape),
                                                            smooth_idf.value<int32_t>(),
                                                            n_samples.value<uint64_t>(),
                                                            volume);
    handle.sync_stream();
  }

  template <legate::Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<!is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::PhysicalStore input,
                  legate::Scalar smooth_idf,
                  legate::Scalar n_samples)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

template <typename idx_t, typename value_t>
__global__ void tfidf_predict_kernel(
  value_t* out, const idx_t* in_cols, const value_t* in_vals, const value_t* in_idf, size_t volume)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = in_vals[idx] * in_idf[in_cols[idx]];
}

template __global__ void tfidf_predict_kernel(
  float*, const uint64_t*, const float*, const float*, size_t);
template __global__ void tfidf_predict_kernel(
  double*, const uint64_t*, const double*, const double*, size_t);

struct tfidf_predict_fn_gpu {
  template <legate::Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::PhysicalStore input_cols,
                  legate::PhysicalStore input_vals,
                  legate::PhysicalStore idf)
  {
    using VAL = legate::type_of_t<CODE>;

    legate_raft::GPUTaskContext gpu_task_context{ctx};
    auto handle = gpu_task_context.handle();
    auto stream = handle.get_stream();

    auto shape = input_cols.shape<DIM>();

    if (shape.empty()) return;

    auto volume = shape.volume();

    int block_size = 256;  // TODO: tune
    int num_blocks = (volume + block_size - 1) / block_size;

    auto input_cols_acc = input_cols.read_accessor<uint64_t, DIM>();
    auto input_vals_acc = input_vals.read_accessor<VAL, DIM>();
    auto idf_acc        = idf.read_accessor<VAL, DIM>();
    auto output_acc     = output.write_accessor<VAL, DIM>();

    tfidf_predict_kernel<<<num_blocks, block_size, 0, stream>>>(output_acc.ptr(shape),
                                                                input_cols_acc.ptr(shape),
                                                                input_vals_acc.ptr(shape),
                                                                idf_acc.ptr(idf.shape<DIM>()),
                                                                volume);
    handle.sync_stream();
  }

  template <legate::Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<!is_supported_gpu<CODE>>* = nullptr>
  void operator()(legate::TaskContext& ctx,
                  legate::PhysicalStore output,
                  legate::PhysicalStore input_cols,
                  legate::PhysicalStore input_vals,
                  legate::PhysicalStore idf)
  {
    LEGATE_ABORT("Code type not supported");
  }
};

}  // namespace

class TfidfFitTask : public Task<TfidfFitTask, TFIDF_FIT> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    auto input      = context.input(0);
    auto output     = context.output(0);
    auto smooth_idf = context.scalar(0);
    auto n_samples  = context.scalar(1);

    assert(input.data().dim() == output.data().dim());
    assert(input.data().code() == output.data().code());

    legate::double_dispatch(input.data().dim(),
                            input.data().code(),
                            tfidf_fit_fn_gpu{},
                            context,
                            output,
                            input,
                            smooth_idf,
                            n_samples);
  }
};

class TfidfPredictTask : public Task<TfidfPredictTask, TFIDF_PREDICT> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    auto input_cols = context.input(0);
    auto input_vals = context.input(1);
    auto idf        = context.input(2);
    auto output     = context.output(0);

    assert(input_cols.data().dim() == input_vals.data().dim() == idf.data().dim() ==
           output.data().dim() == 1);

    legate::double_dispatch(input_cols.data().dim(),
                            input_vals.data().code(),
                            tfidf_predict_fn_gpu{},
                            context,
                            output,
                            input_cols,
                            input_vals,
                            idf);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::TfidfFitTask::register_variants();
  legate_raft::TfidfPredictTask::register_variants();
}

}  // namespace
