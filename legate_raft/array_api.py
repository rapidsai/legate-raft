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
from typing import Union

import legate.core.types as ty
import numpy as np
from legate.core import LogicalStore, get_legate_runtime

from legate_raft.cffi import OpCode
from legate_raft.core import _determine_dtype
from legate_raft.library import user_context as library

legate_runtime = get_legate_runtime()


def _determine_dtype_from_scalar(value) -> ty.Type:
    try:
        return _determine_dtype(getattr(value, "type", getattr(value, "dtype")))
    except AttributeError:
        return _determine_dtype(np.asanyarray(value).dtype)


def fill(shape, fill_value, dtype=None) -> LogicalStore:
    if dtype is None:
        dtype = _determine_dtype_from_scalar(fill_value)

    if isinstance(shape, int):
        shape = (shape,)

    result = legate_runtime.create_array(
        dtype, shape, optimize_scalar=(shape == tuple())
    )
    assert result.type == dtype

    task = legate_runtime.create_auto_task(library, OpCode.FILL)
    task.add_output(result)
    task.add_scalar_arg(fill_value, result.type)
    task.execute()

    return result.data


def srange(start, stop=None, step=None, dtype=None) -> LogicalStore:
    if stop is None:
        stop = start
        start = type(stop)(0)
    if step is None:
        step = type(stop)(1)

    assert type(start) == type(stop) == type(step)

    if dtype is None:
        try:
            dtype = stop.dtype
        except AttributeError:
            stop = np.asanyarray(stop)
            dtype = stop.dtype

    size = (stop - start) // step
    shape = (size,)

    result = legate_runtime.create_store(dtype, shape, optimize_scalar=(shape == (1,)))
    assert result.type == dtype

    task = legate_runtime.create_auto_task(library, OpCode.RANGE)
    task.add_output(result)
    task.add_scalar_arg(start, result.type)
    task.add_scalar_arg(step, result.type)
    task.execute()

    return result


def _sanitize_axis(axis: int, ndim: int) -> int:
    sanitized = axis
    if sanitized < 0:
        sanitized += ndim
    if sanitized < 0 or sanitized >= ndim:
        raise ValueError(f"Invalid axis {axis} for a {ndim}-D store")
    return sanitized


def sum_over_axis(input: LogicalStore, axis: int) -> LogicalStore:
    """
    Sum values along the chosen axis
    Parameters
    ----------
    input : LogicalStore
        Input to sum
    axis : int
        Axis along which the summation should be done
    Returns
    -------
    LogicalStore
        Summation result
    """
    sanitized = _sanitize_axis(axis, input.ndim)

    # Compute the output shape by removing the axis being summed over
    res_shape = tuple(ext for dim, ext in enumerate(input.shape) if dim != sanitized)
    result = fill(res_shape, 0, dtype=input.type)

    # Broadcast the output along the contracting dimension
    promoted = result.promote(axis, input.shape[axis])

    assert promoted.shape == input.shape

    task = legate_runtime.create_auto_task(library, OpCode.SUM_OVER_AXIS)
    task.add_input(input)
    task.add_reduction(promoted, ty.ReductionOpKind.ADD)
    task.add_alignment(input, promoted)

    task.execute()

    return result


def _add_constant(input: LogicalStore, value: Number) -> LogicalStore:
    result = legate_runtime.create_store(input.type, input.shape)

    task = legate_runtime.create_auto_task(library, OpCode.ADD_CONSTANT)
    task.add_input(input)
    task.add_scalar_arg(value, input.type)
    task.add_output(result)
    task.add_alignment(input, result)

    task.execute()

    return result


def log(input: Union[LogicalStore, Number]) -> Union[LogicalStore, Number]:
    if isinstance(input, Number):
        return math.log(input)

    result = legate_runtime.create_store(input.type, input.shape)

    task = legate_runtime.create_auto_task(library, OpCode.LOG)
    task.add_input(input)
    task.add_output(result)
    task.add_alignment(input, result)

    task.execute()

    return result


def _add_stores(x1: LogicalStore, x2: LogicalStore) -> LogicalStore:
    assert x1.type == x2.type

    result = legate_runtime.create_store(x1.type, x1.shape)

    task = legate_runtime.create_auto_task(library, OpCode.ADD)
    task.add_input(x1)
    task.add_input(x2)
    task.add_output(result)
    task.add_alignment(x1, x2)
    task.add_alignment(x1, result)

    task.execute()

    return result


def _add_broadcast(x1: LogicalStore, x2: LogicalStore) -> LogicalStore:
    assert x1.type == x2.type

    # All promotion by inserting starting, it may be that other dimensions
    # have to match exactly right now.
    for i in range(x2.ndim - x1.ndim):
        x1 = x1.promote(i, x2.shape[i])
    for i in range(x1.ndim - x2.ndim):
        x2 = x2.promote(i, x1.shape[i])

    result = legate_runtime.create_store(x1.type, x1.shape)
    task = legate_runtime.create_auto_task(library, OpCode.ADD)
    task.add_input(x1)
    task.add_input(x2)
    task.add_alignment(x1, x2)
    task.add_output(result)
    task.add_alignment(x1, result)

    task.execute()

    return result


def add(
    x1: Union[LogicalStore, Number], x2: Union[LogicalStore, Number]
) -> Union[LogicalStore, Number]:
    if isinstance(x1, Number):
        if isinstance(x2, Number):
            return x1 + x2  # native function
        else:
            return add(x2, x1)  # swap operands

    elif isinstance(x2, Number):
        return _add_constant(x1, x2)
    elif x1.shape == x2.shape:
        return _add_stores(x1, x2)
    else:
        return _add_broadcast(x1, x2)


def negative(lhs: LogicalStore) -> LogicalStore:
    return multiply(lhs, -1.0)


def subtract(
    x1: Union[LogicalStore, Number], x2: Union[LogicalStore, Number]
) -> Union[LogicalStore, Number]:
    if isinstance(x1, Number) and isinstance(x2, Number):
        return x1 - x2  # native function
    else:
        return add(x1, negative(x2))


def multiply(input: LogicalStore, value: Number) -> LogicalStore:
    assert isinstance(input, LogicalStore)
    assert isinstance(value, Number)

    result = legate_runtime.create_store(input.type, input.shape)

    task = legate_runtime.create_auto_task(library, OpCode.MULTIPLY_BY_CONSTANT)
    task.add_input(input)
    task.add_scalar_arg(value, input.type)
    task.add_output(result)
    task.add_alignment(input, result)

    task.execute()

    return result


def power(input: LogicalStore, value: Number) -> LogicalStore:
    assert isinstance(input, LogicalStore)
    assert isinstance(value, Number)

    result = legate_runtime.create_store(input.type, input.shape)

    task = legate_runtime.create_auto_task(library, OpCode.POWER)
    task.add_input(input)
    task.add_scalar_arg(value, input.type)
    task.add_output(result)
    task.add_alignment(input, result)

    task.execute()

    return result


def max(x: LogicalStore, axis: int) -> Number:
    sanitized = _sanitize_axis(axis, x.ndim)

    try:
        limit_min = np.finfo(x.type.type.to_pandas_dtype()).min
    except ValueError:
        limit_min = np.iinfo(x.type.type.to_pandas_dtype()).min

    res_shape = tuple(ext for dim, ext in enumerate(x.shape) if dim != sanitized)
    result = fill(res_shape, limit_min, x.type)

    promoted = result.promote(axis, x.shape[axis])
    assert promoted.shape == x.shape

    task = legate_runtime.create_auto_task(library, OpCode.FIND_MAX)
    task.add_input(x)
    task.add_reduction(promoted, ty.ReductionOpKind.MAX)
    task.add_alignment(x, promoted)

    task.execute()

    return result


def unique(input: LogicalStore, radix: int = 8) -> LogicalStore:
    """
    Finds unique elements in the input and returns them in a store

    Parameters
    ----------
    input : LogicalStore
        Input

    Returns
    -------
    LogicalStore
        Result that contains only the unique elements of the input
    """

    if input.ndim > 1:
        raise ValueError("`unique` accepts only 1D stores")

    # Create an unbound store to collect local results
    result = legate_runtime.create_store(input.type, shape=None, ndim=1)

    task = legate_runtime.create_auto_task(library, OpCode.UNIQUE)
    task.add_input(input)
    task.add_output(result)

    task.execute()

    # Perform global reduction using a reduction tree
    return legate_runtime.tree_reduce(library, OpCode.UNIQUE, result, radix=radix)
