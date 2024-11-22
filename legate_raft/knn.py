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

import legate.core.types as types
import numpy as np
from legate.core import get_legate_runtime

from .core import get_logical_array
from .library import user_context as library
from .library import user_lib
from .utils import _track_provenance

legate_runtime = get_legate_runtime()


@_track_provenance
def run_knn(
    index: np.ndarray, search: np.ndarray, n_neighbors: int, metric: str = "l2"
):
    """
    Run a brute-force kNN.

    Parameters
    ----------
    index : legate store like
        The index to search the nearest neighbors in.  This data should be
        large and may be distributed over multiple nodes.
    search : legate store like
        The search query to find the nearest neighbors for.  Right now, this
        must be small enough to be broadcast on (copied to) all workers.
    n_neighbors : int
        The number of neighbors to find.
    metric : str
        The metric to use, currently supports "L2".

    Returns
    -------
    result_dist : legate store
        A legate store with the distance to each nearest neighbors.
    result_indices : legate store
        The indices of the nearest neighbors for each point in `search`.

    """
    index = get_logical_array(index)
    search = get_logical_array(search)

    if index.type != search.type:
        raise TypeError("kNN index and search type must match.")
    if index.type == types.float32:
        dtype = types.float32
    elif index.type == types.float64:
        dtype = types.float64
    else:
        # Raise a slightly nicer error than C might
        raise TypeError("kNN only supports float32 and float64.")

    if index.ndim != 2 or search.ndim != 2:
        raise ValueError("kNN index and search must be two dimensional.")
    if index.shape[1] != search.shape[1]:
        raise ValueError("kNN index and search dimensionality mismatch.")

    # Create late-bound result stores (just to not deal with broadcasting)
    result_dist = legate_runtime.create_store(dtype, ndim=2)
    result_indices = legate_runtime.create_store(types.int64, ndim=2)

    # Run KMeans Fit task
    knn_task = legate_runtime.create_auto_task(library, user_lib.cffi.RAFT_KNN)
    knn_task.add_scalar_arg(n_neighbors, types.int64)
    knn_task.add_scalar_arg(metric.lower(), types.string_type)

    knn_task.add_input(index)
    knn_task.add_input(search)
    knn_task.add_output(result_indices)
    knn_task.add_output(result_dist)
    # Must not split the second dimension and must broadcast `search` fully:
    knn_task.add_broadcast(index, (1,))
    knn_task.add_broadcast(search, (0, 1))

    knn_task.add_nccl_communicator()
    knn_task.execute()

    return result_dist, result_indices
