# Copyright 2023-2024 NVIDIA Corporation
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
from legate.core import get_legate_runtime

from legate_raft.core import as_array, as_store, get_logical_array
from legate_raft.library import user_context as library
from legate_raft.library import user_lib
from legate_raft.utils import _track_provenance

legate_runtime = get_legate_runtime()


class KMeans:
    """k-Means clustering, compare also :py:class:`sklearn.cluster.KMeans`.

    This version of k-Means currently always uses the default parameters
    initializing with a single ``KMmeans||``.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters.
    """

    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters

        # Currently cluster_centers is the NumPy version, we also hold on
        # to the legate store:
        self._cluster_centers_lg = None
        self._cluster_centers_ = None
        self._n_iter_ = None

    @_track_provenance
    def fit(self, X):
        """Find the k-means clustering for `X`."""
        X_arr = get_logical_array(X)

        if X_arr.type == types.float32:
            X_dtype = types.float32
        elif X_arr.type == types.float64:
            X_dtype = types.float64
        else:
            # Raise a slightly nicer error than C might
            raise TypeError("KMeans only supports float32 and float64.")

        if X_arr.ndim != 2:
            raise ValueError("X must be two dimensional.")

        # Add a store which will be filled with the result centroids
        centroids_store = legate_runtime.create_store(
            X_dtype, shape=(self.n_clusters, X_arr.shape[1])
        )
        # And an additional one to fill with information (currently n_iter)
        # TODO: This is 2-D because 1-d didn't seem to work, it probably could
        #       be scalar.
        info_store = legate_runtime.create_store(X_dtype, ndim=2)

        # Run KMeans Fit task
        kmeans_fit_task = legate_runtime.create_auto_task(
            library, user_lib.cffi.RAFT_KMEANS_FIT
        )

        # NOTE: The configuration is order dependent
        kmeans_fit_task.add_input(X_arr)
        kmeans_fit_task.add_output(centroids_store)
        kmeans_fit_task.add_output(info_store)

        # X must not be split along dimension 1 (the second).
        kmeans_fit_task.add_broadcast(X_arr, (1,))
        # Centroids is replicated on each node/shard:
        kmeans_fit_task.add_broadcast(centroids_store, (0, 1))

        kmeans_fit_task.add_nccl_communicator()
        kmeans_fit_task.execute()

        # We only store the legate stores as unpacking them is blocking
        self._cluster_centers_lg = centroids_store
        self._info_store = info_store

        return self

    @_track_provenance
    def predict(self, X):
        """Predict the closest cluster for each sample."""
        if self._cluster_centers_lg is None:
            raise RuntimeError("KMeans appears to be not yet fit.")

        X_arr = get_logical_array(X)

        if X_arr.ndim != 2:
            raise ValueError("X must be two dimensional.")
        if X_arr.shape[1] != self._cluster_centers_lg.shape[1]:
            raise ValueError("X features do not match those of centers.")

        if X_arr.type != self._cluster_centers_lg.type:
            # Enforce the same type for now, we could relax this by casting
            # either the centroids or X.
            raise TypeError(
                f"X must have the same dtype as centers {self._cluster_centers_lg.type}"
                " but is of type {X_arr.type}."
            )

        if X_arr.ndim != 2:
            raise ValueError("X must be two dimensional.")

        predict_task = legate_runtime.create_auto_task(
            library, user_lib.cffi.RAFT_KMEANS_PREDICT
        )

        labels = legate_runtime.create_store(types.int32, shape=(X.shape[0],))
        # Insert dimension into labels to align it with X (is there an easier way?)
        labels_prom = labels.promote(1, X.shape[1])

        predict_task.add_input(X_arr)
        predict_task.add_input(self._cluster_centers_lg)
        predict_task.add_output(labels_prom)
        # X and labels are aligned and not split along last dim
        # while the centroids store is replicated to all nodes/tasks.
        predict_task.add_broadcast(X_arr, (1,))
        predict_task.add_broadcast(labels_prom, (1,))
        predict_task.add_broadcast(self._cluster_centers_lg, (0, 1))
        predict_task.add_alignment(X_arr, labels_prom)

        predict_task.execute()

        return labels

    @property
    def cluster_centers_(self):
        """NumPy array containing the cluster centers."""
        # Property as the conversion from legate is blocking
        if self._cluster_centers_ is not None:
            return self._cluster_centers_
        if self._cluster_centers_lg is None:
            return None
        self._cluster_centers_ = as_array(self._cluster_centers_lg)
        return self._cluster_centers_

    @property
    def n_iter_(self):
        """Number of iteration to converge after initialization."""
        # Property as the conversion from legate is blocking
        if self._n_iter_ is not None:
            return self._n_iter_
        if self._info_store is None:
            return None
        # n_iter is currently stored as a float, but this is presumably OK
        self._n_iter_ = int(as_array(self._info_store)[0, 0])
        return self._n_iter_

    def __getstate__(self):
        return dict(
            n_iter=self.n_iter_,
            cluster_centers=self.cluster_centers_,
            n_clusters=self.n_clusters,
        )

    def __setstate__(self, state):
        self.n_clusters = state["n_clusters"]
        self._n_iter_ = state["n_iter"]
        self._cluster_centers_ = state["cluster_centers"]
        # Also restore the legate centers used when predicting:
        self._cluster_centers_lg = as_store(self.cluster_centers_)
