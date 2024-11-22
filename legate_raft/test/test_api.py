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

import numpy as np
import pytest
from hypothesis import example, given, note, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays, from_dtype
from legate.core import get_legate_runtime
from numpy.testing import assert_array_almost_equal_nulp, assert_array_equal
from scipy.sparse import coo_array

from legate_raft import argmax, fill, log, sum_over_axis
from legate_raft.core import as_array, as_store
from legate_raft.sparse import COOStore

legate_runtime = get_legate_runtime()


@example(shape=(3, 4), fill_value=1.0)
@given(
    shape=array_shapes(),
    fill_value=st.sampled_from(["i4", "i8", "f4", "f8"])
    .map(np.dtype)
    .flatmap(from_dtype),
)
@settings(deadline=None, max_examples=1000)
def test_fill(shape, fill_value):
    store = fill(shape, fill_value)
    assert store.shape == shape
    array = as_array(store)
    array_copy = array.copy()
    array_copy.fill(fill_value)
    assert_array_equal(array, array_copy)


@settings(deadline=None)
@given(
    A=arrays(
        dtype=np.float64,
        elements=from_dtype(
            np.dtype("f"),
            allow_nan=False,
            allow_infinity=False,
            min_value=0.009999999776482582,
        ),
        shape=array_shapes(),
        # shape=array_shapes(min_dims=2, max_dims=2, min_side=4, max_side=20),
    )
)
def test_log(A):
    result_np = np.log(A)
    result_lg = as_array(log(as_store(A)))
    note(f"A: {A}")
    note(f"np: {result_np}")
    note(f"lg: {result_lg}")
    assert_array_almost_equal_nulp(result_np, result_lg)


@settings(deadline=None)
@given(
    A=arrays(
        dtype=np.float64,
        elements=from_dtype(np.dtype("f"), allow_nan=False, allow_infinity=False),
        shape=(10, 10),
    )
)
def test_sum_over_axis(A):
    result_np = np.sum(A, axis=1)
    result_lg = as_array(sum_over_axis(as_store(A), axis=1))
    note(f"A: {A}")
    note(f"np: {result_np}")
    note(f"lg: {result_lg}")
    # Hypothesis generates awkward cases, including some that will show
    # catastrophic anihiliation and NumPy pairwise summation result differ
    # a lot. Comparing with the maximum abs value as rtol may be stable enough.
    rtol = np.max(abs(A), axis=1) * np.finfo(A.dtype).eps * 4
    # Assert allclose doesn't like an `rtol` vector it seems.
    assert result_np.shape == result_lg.shape
    assert np.allclose(result_np, result_lg, rtol=rtol)


@settings(deadline=None)
@given(
    A=arrays(
        dtype=np.float32,
        elements=from_dtype(np.dtype("f"), allow_nan=False),
        shape=array_shapes(min_dims=2, max_dims=2, min_side=4, max_side=20),
    )
)
def test_argmax(A):
    assert A.ndim == 2
    result_np = np.argmax(A, axis=1)
    result_lg = as_array(argmax(as_store(A), axis=1))
    assert_array_equal(result_np, result_lg)


@pytest.mark.xfail(reason="The coo_mm implementation is currently broken.")
def test_coo_matmat():
    A = coo_array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 2, 3, 0, 0],
            [0, 0, 0, 4, 0],
        ],
        dtype=np.float32,
    )

    B = np.array(
        [[1, 0, 0, 0], [0, 2, 3, 0], [0, 0, 0, 4], [0, 0, 0, 5], [0, 0, 0, 6]],
        dtype=np.float32,
    )

    C = A @ B
    A_store = COOStore.from_sparse_array(A)
    C_store = A_store @ as_store(B)

    assert_array_almost_equal_nulp(C, as_array(C_store))


@settings(deadline=None)
@given(
    A=arrays(
        dtype=np.float32,
        elements=from_dtype(np.dtype("f"), allow_nan=False),
        shape=array_shapes(min_dims=2, max_dims=2, min_side=4, max_side=20),
    )
)
def test_coo_len(A):
    A_coo = COOStore.from_sparse_array(coo_array(A))
    assert len(A_coo) == len(A)
