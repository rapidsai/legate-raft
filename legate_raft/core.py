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

import math
from numbers import Number
from typing import Any, Union

import numpy as np
from legate.core import LogicalArray, LogicalStore, get_legate_runtime
from legate.core import types as ty

from .cffi import OpCode
from .library import user_context as library

# from legate.core._legion.future import Future

legate_runtime = get_legate_runtime()


_NP2LT_TYPES = {
    np.dtype(np.bool_): ty.bool_,
    np.dtype(np.int8): ty.int8,
    np.dtype(np.int16): ty.int16,
    np.dtype(np.int32): ty.int32,
    np.dtype(np.int64): ty.int64,
    np.dtype(np.uint8): ty.uint8,
    np.dtype(np.uint16): ty.uint16,
    np.dtype(np.uint32): ty.uint32,
    np.dtype(np.uint64): ty.uint64,
    np.dtype(np.float16): ty.float16,
    np.dtype(np.float32): ty.float32,
    np.dtype(np.float64): ty.float64,
    np.dtype(np.complex64): ty.complex64,
    np.dtype(np.complex128): ty.complex128,
    np.dtype(np.str_): ty.string_type,
}


def create_matrix(n_rows, n_cols, dtype, n_parts=None):
    """
    Create a two dimensional legate store, if `n_parts` is given,
    partitions it into as many parts along both dimensions.
    """
    store = legate_runtime.create_store(
        _NP2LT_TYPES[dtype],
        shape=(n_rows, n_cols),
        optimize_scalar=False,
    )

    if n_parts is None:
        return store

    n_rows_per_parts = math.ceil(n_rows / n_parts)

    return store.partition_by_tiling((n_rows_per_parts, n_cols)).store()


def create_vector(n_rows, dtype, n_parts=None):
    """
    Create a one dimensional legate store, if `n_parts` is given,
    partitions it into as many parts.
    """
    store = legate_runtime.create_store(
        _NP2LT_TYPES[dtype],
        shape=(n_rows,),
        optimize_scalar=False,
    )

    if n_parts is None:
        return store

    n_rows_per_parts = math.ceil(n_rows / n_parts)

    return store.partition_by_tiling((n_rows_per_parts,)).store()


def as_store(array: np.ndarray) -> LogicalStore:
    if isinstance(array, LogicalStore):
        return array

    return legate_runtime.create_store_from_buffer(
        dtype=_NP2LT_TYPES[array.dtype], shape=array.shape, data=array, read_only=False
    )


def as_array(store: LogicalStore) -> np.ndarray:
    return np.asarray(store.get_physical_store().get_inline_allocation())


def as_scalar(store: LogicalStore) -> Number:
    array = as_array(store)
    assert array.ndim == 1 and array.shape == (1,)
    return array.item()


_NativeLegateType = ty.Type
DataType = Union[type, np.dtype, _NativeLegateType]


def _determine_dtype(dtype: DataType) -> ty.Type:
    if type(dtype) is ty.Type:
        return dtype
    elif dtype is int:
        return ty.int64
    elif dtype is float:
        return ty.float64
    elif dtype is bool:
        return ty.bool_
    else:
        try:
            return _NP2LT_TYPES[dtype]
        except KeyError:
            raise ValueError(f"Unsupported dtype: {dtype} ({type(dtype)})")


def convert(input: LogicalStore, dtype: DataType) -> LogicalStore:
    target_dtype = _determine_dtype(dtype)
    if _determine_dtype(input.type) == target_dtype:
        return input

    result = legate_runtime.create_store(target_dtype, input.shape)
    task = legate_runtime.create_auto_task(library, OpCode.CONVERT)
    task.add_input(input)
    task.add_output(result)
    task.add_alignment(input, result)
    task.execute()

    return result


def copy(input: LogicalStore) -> LogicalStore:
    result = legate_runtime.create_store(input.type, input.shape)
    task = legate_runtime.create_auto_task(library, OpCode.COPY)
    task.add_input(input)
    task.add_output(result)
    task.add_alignment(input, result)
    task.execute()

    return result


def to_scalar(input: LogicalStore) -> Number:
    """Extracts a Python scalar value from a Legate store
       encapsulating a single scalar

    Args:
        input (LogicalStore): The Legate store encapsulating a scalar

    Returns:
        number: A Python scalar
    """
    # This operation blocks until the data in the LogicalStore
    # is available and correct
    return as_array(input)[0]


def get_logical_array(obj: Any) -> LogicalArray:
    """
    Get a logical array from an object supporting the legate data interface.

    .. note::
        This version currently rejects nullable arrays.
    """
    if isinstance(obj, LogicalArray):
        return obj

    # Extract the very first store and make sure it is the only one
    arrays = obj.__legate_data_interface__["data"].values()
    if len(arrays) != 1:
        raise ValueError("object must expose a single logical array")

    array = next(iter(arrays))
    if array.nullable:
        raise ValueError("Array must not be nullable.")

    return array
