# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from math import floor, log2

import cupy as cp
from legate.core import get_legate_runtime
from legate.core import types as ty

from legate_raft.library import user_context as library
from legate_raft.library import user_lib
from legate_raft.utils import _track_provenance

# TODO (csa) There is a chance that the algorithm as implemented is skweing
# things to lower values.

legate_runtime = get_legate_runtime()


def _compute_batch_size(available_memory, density, size):
    "Approximate optimal batch size for given memory and density."
    required_memory = 2 * density * size
    return floor(log2(available_memory / required_memory))


@_track_provenance
def make_rmat(
    r_scale=10,
    c_scale=10,
    random_seed=0,
    *,
    density=0.001,
    a=0.55,
    b=0.15,
    c=0.15,
    dtype=ty.uint64,
    batch_size=None,
):
    assert dtype == ty.uint64  # currently only implemented for uint64

    if batch_size is None:
        try:
            batch_size = _compute_batch_size(
                0.1 * cp.cuda.Device().mem_info[1], density, dtype.size
            )
        except RuntimeError:  # use relatively safe default
            batch_size = 38

    scale = r_scale + c_scale
    n_partitions = max(1, 2 ** (scale - batch_size))
    r_scale_local = r_scale - int(log2(n_partitions))
    assert 2 ** (r_scale - r_scale_local) == n_partitions
    nnz_max = int(density * 2 ** (r_scale_local + c_scale))

    assert n_partitions > 0
    assert r_scale_local > 0

    out_src = legate_runtime.create_store(dtype)
    out_dst = legate_runtime.create_store(dtype)

    task = legate_runtime.create_manual_task(
        library,
        user_lib.cffi.MAKE_RMAT,
        ((n_partitions,)),
    )
    task.add_scalar_arg(random_seed, ty.uint64)
    task.add_scalar_arg(r_scale_local, ty.uint64)
    task.add_scalar_arg(c_scale, ty.uint64)
    task.add_scalar_arg(nnz_max, ty.uint64)
    task.add_scalar_arg(a, ty.float32)
    task.add_scalar_arg(b, ty.float32)
    task.add_scalar_arg(c, ty.float32)
    task.add_output(out_src)
    task.add_output(out_dst)
    task.execute()

    assert out_src.shape[0] == out_dst.shape[0]
    shape = 2**r_scale, 2**c_scale

    return shape, out_src, out_dst
