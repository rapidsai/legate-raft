# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import legate.core.types as ty
from legate.core import LogicalStore, get_legate_runtime
from legate.core.types import Type

from legate_raft.array_api import fill
from legate_raft.array_api import max as lg_max
from legate_raft.cffi import OpCode
from legate_raft.core import as_scalar
from legate_raft.library import user_context as library

legate_runtime = get_legate_runtime()


def multiply(rhs1: LogicalStore, rhs2: LogicalStore) -> LogicalStore:
    if rhs1.type != rhs2.type or rhs1.shape != rhs2.shape:
        raise ValueError("Stores to add must have the same type and shape")

    result = legate_runtime.create_store(rhs1.type.type, rhs1.shape)

    task = legate_runtime.create_auto_task(library, OpCode.MUL)
    task.add_input(rhs1)
    task.add_input(rhs2)
    task.add_output(result)
    task.add_alignment(result, rhs1)
    task.add_alignment(result, rhs2)

    task.execute()

    return result


def bincount(
    input: LogicalStore,
    num_bins: Union[int, None] = None,
    output_type: Type = ty.uint64,
) -> LogicalStore:
    """
    Counts the occurrences of each bin index
    Parameters
    ----------
    input : LogicalStore
        Input to bin-count
    num_bins : int
        Number of bins
    output_type: Type
        The type of the output array.
        Default=uint64

    Returns
    -------
    LogicalStore
        Counting result
    """
    assert input.type in (ty.int32, ty.int64, ty.uint32, ty.uint64)

    if num_bins is None:
        num_bins = as_scalar(lg_max(input, axis=0)) + 1

    result = fill((num_bins,), 0, output_type)

    task = legate_runtime.create_auto_task(library, OpCode.BINCOUNT)
    task.add_input(input)
    # Broadcast the result store. This commands the Legate runtime to give
    # the entire store to every task instantiated by this task descriptor
    task.add_broadcast(result)
    # Declares that the tasks will do reductions to the result store and
    # that outputs from the tasks should be combined by addition
    task.add_reduction(result, ty.ReductionOpKind.ADD)

    task.execute()

    return result
