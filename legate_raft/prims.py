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

from legate.core import get_legate_runtime
from legate.core import types as ty

from .array_api import fill, unique
from .cffi import OpCode
from .core import LogicalStore, convert
from .library import user_context as library
from .multiarray import bincount
from .sparse import SparseStore

legate_runtime = get_legate_runtime()


def map_labels(labels: LogicalStore, classes: LogicalStore) -> LogicalStore:
    result = legate_runtime.create_store(labels.type, shape=labels.shape)

    task = legate_runtime.create_auto_task(library, OpCode.MAP_LABELS)
    task.add_input(labels)
    task.add_input(classes)
    task.add_broadcast(classes)
    task.add_output(result)
    task.add_alignment(labels, result)
    task.execute()

    return result


def invert_labels(labels: LogicalStore, classes: LogicalStore) -> LogicalStore:
    """
    Takes a set of labels that have been mapped to be drawn
    from a monotonically increasing set and inverts them to
    back to the original set of classes.

    Parameters
    ----------

    labels : array-like of size (n,) labels to invert
    classes : array-like of size (n_classes,) the unique set
              of classes for inversion. It is assumed that
              the classes are ordered by their corresponding
              monotonically increasing label.
    copy : boolean if true, a copy will be returned and the
           operation will not be done in place.

    Returns
    -------

    inverted labels : array-like of size (n,)
    """
    assert labels.type == classes.type
    result = legate_runtime.create_store(labels.type, shape=labels.shape)

    task = legate_runtime.create_auto_task(library, OpCode.INVERT_LABELS)
    task.add_input(labels)
    task.add_input(classes)
    task.add_broadcast(classes)
    task.add_output(result)
    task.add_alignment(labels, result)
    task.execute()

    return result


def make_monotonic(labels: LogicalStore) -> tuple[LogicalStore, LogicalStore]:
    """
    Takes a set of labels that might not be drawn from the
    set [0, n-1] and renumbers them to be drawn that
    interval.

    Replaces labels not present in classes by len(classes)+1.

    Parameters
    ----------

    labels : array-like of size (n,) labels to convert
    classes : array-like of size (n_classes,) the unique
              set of classes in the set of labels

    Returns
    -------

    mapped_labels : array-like of size (n,)
    classes : array-like of size (n_classes,)
    """
    classes = unique(labels)
    mapped_labels = map_labels(labels, classes)
    return mapped_labels, classes


def count_features(X: SparseStore, Y: LogicalStore, n_classes: int) -> LogicalStore:
    assert X.shape[0] == Y.shape[0]
    assert X.row.type == X.col.type == ty.uint64
    assert Y.type == ty.int64

    n_features = X.shape[1]
    output_shape = (n_classes, n_features)
    result = fill(output_shape, 0, ty.float64)

    n_rows, n_cols = X.shape

    task = legate_runtime.create_auto_task(library, OpCode.COUNT_FEATURES)
    task.add_input(convert(X.data, ty.float64))
    task.add_input(X.row)
    task.add_input(X.col)
    task.add_input(Y)
    task.add_alignment(X.data, X.row)
    task.add_alignment(X.data, X.col)
    task.add_scalar_arg(n_classes, ty.uint64)
    task.add_scalar_arg(n_rows, ty.uint64)
    task.add_scalar_arg(n_cols, ty.uint64)
    task.add_scalar_arg(n_features, ty.uint64)
    # TODO: Replace with better constraints.
    task.add_broadcast(Y)
    task.add_reduction(result, ty.ReductionOpKind.ADD)
    task.add_broadcast(result)

    task.execute()
    return result


def count_classes(Y: LogicalStore, n_classes: int):
    return bincount(Y, num_bins=n_classes)
