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

import warnings
from dataclasses import dataclass
from typing import Union

from legate.core import LogicalStore, get_legate_runtime
from legate.core import types as ty
from scipy.sparse import coo_array, coo_matrix, csr_array, csr_matrix

from legate_raft.core import as_store
from legate_raft.core import copy as copy_store

from ..array_api import fill
from ..cffi import OpCode
from ..core import as_array, convert
from ..library import user_context as library

SparseArray = Union[csr_array, csr_matrix, coo_array, coo_matrix]


legate_runtime = get_legate_runtime()


@dataclass
class COOStore:
    data: LogicalStore
    row: LogicalStore
    col: LogicalStore

    shape: tuple[int]

    @classmethod
    def from_sparse_array(cls, array: SparseArray, *, index_type=None) -> "COOStore":
        if index_type is None:
            index_type = ty.uint64

        if isinstance(array, cls):
            assert array.row.type == index_type
            return array

        coo = array.tocoo()
        ret = cls(
            data=as_store(coo.data),
            row=convert(as_store(coo.row), index_type),
            col=convert(as_store(coo.col), index_type),
            shape=coo.shape,
        )
        return ret

    @classmethod
    def unbound(cls, shape, dtype, index_type=ty.uint64) -> "COOStore":
        ret = cls(
            data=legate_runtime.create_store(dtype, shape=None, ndim=1),
            row=legate_runtime.create_store(index_type, shape=None, ndim=1),
            col=legate_runtime.create_store(index_type, shape=None, ndim=1),
            shape=shape,
        )
        return ret

    def to_sparse_array(self) -> coo_array:
        return coo_array(
            (as_array(self.data), (as_array(self.row), as_array(self.col))),
            shape=self.shape,
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def type(self):
        return self.data.type

    def to_type(self, type_):
        return self.__class__(
            data=convert(self.data, type_),
            row=self.row,
            col=self.col,
            shape=self.shape,
        )

    def partition_by_tiling(self, rows_per_partition):
        return PartitionedCOOStore(
            data=self.data.partition_by_tiling((rows_per_partition,)),
            row=self.row.partition_by_tiling((rows_per_partition,)),
            col=self.col.partition_by_tiling((rows_per_partition,)),
            shape=self.shape,
        )

    @property
    def nnz(self):
        assert self.row.shape == self.col.shape == self.data.shape
        return self.data.shape[0]

    def __matmul__(self, other):
        if isinstance(other, LogicalStore):
            return _coo_mm(self, other)
        else:
            raise NotImplementedError(
                f"Matrix multiplication for type {type(other)} is not supported."
            )

    def __len__(self):
        if self.ndim > 0:
            return self.shape[0]
        else:
            raise TypeError("len() of unsized object")


@dataclass
class PartitionedCOOStore:
    data: LogicalStore
    row: LogicalStore
    col: LogicalStore

    shape: tuple[int]

    @property
    def type(self):
        return self.data.store().type

    def __len__(self):
        if self.ndim > 0:
            return self.shape[0]
        else:
            raise TypeError("len() of unsized object")


SparseStore = COOStore


def as_sparse_store(array: SparseArray) -> SparseStore:
    return COOStore.from_sparse_array(array)


def _coo_mm(A: COOStore, B: LogicalStore) -> LogicalStore:
    raise NotImplementedError("The implementation appears to be broken at the moment.")

    # if B.transformed and not legate_runtime.runtime.machine.only(ProcessorKind.CPU):
    if B.transformed:
        # Need to create copy of transposed matrix, since the matrix operand in
        # A @ B must be contiguous for the non-CPU task variants.
        warnings.warn(
            "Creating copy of operand B for matrix multiplication "
            "as it must be contiguous.",
            UserWarning,
        )
        B = copy_store(B)

    assert A.type == B.type

    m, k = A.shape
    k_, n = B.shape
    assert k == k_
    result_shape = (m, n)
    print("mkn", m, k, n)

    assert A.type in (ty.float32, ty.float64)
    C = fill(result_shape, 0, A.type)
    A_row_promoted = A.row.promote(1, result_shape[1])
    A_col_promoted = A.col.promote(1, result_shape[1])
    A_data_promoted = A.data.promote(1, result_shape[1])

    task = legate_runtime.create_auto_task(library, OpCode.SPARSE_COO_MM)
    # This task is only implemented for these index types.
    assert A.row.type == A.col.type == ty.uint64

    # Add inputs and outputs
    print(A_row_promoted.shape, C.shape, A.col.shape, A.data.shape, B.shape)
    task.add_reduction(C, ty.ReductionOpKind.ADD)
    task.add_input(A_row_promoted)
    task.add_input(A_col_promoted)
    task.add_input(A_data_promoted)
    task.add_input(B)

    # Partitioning
    task.add_broadcast(B)
    task.add_broadcast(A_row_promoted, 1)
    task.add_alignment(A_row_promoted, A_col_promoted)
    task.add_alignment(A_row_promoted, A_data_promoted)
    task.add_alignment(A_row_promoted, C)

    # Scalars
    task.add_scalar_arg(m, ty.int32)
    task.add_scalar_arg(k, ty.int32)
    task.add_scalar_arg(n, ty.int32)
    task.add_scalar_arg(A.nnz, ty.uint64)

    task.execute()

    return C
