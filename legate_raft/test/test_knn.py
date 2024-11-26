# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupynumeric as cn
import numpy as np
import pytest

# Use sklearn make_blobs for now, as legate-raft one repeated values
# see https://github.com/rapidsai/raft/issues/1127
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

from legate_raft import run_knn


@pytest.mark.parametrize(
    "n_index_rows, n_features, k, dtype",
    [
        (5000, 20, 8, "float64"),
        (10000, 10, 20, "float32"),
    ],
)
def test_knn(n_index_rows, n_features, k, dtype):
    n_search_rows = 33
    metric = "l2"

    X, _ = make_blobs(n_index_rows + n_search_rows, n_features, centers=5)
    # cupynumeric makes casting easier and allows using `np.asarray`
    X = cn.asarray(X).astype(np.float64)
    index = X[:n_index_rows, :]
    search = X[n_index_rows:, :]

    index_arr = np.asarray(index)
    search_arr = np.asarray(search)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(index_arr)
    ref_distances, ref_indices = nn.kneighbors(search_arr, return_distance=True)

    distances, indices = run_knn(index, search, k, metric)
    distances = np.asarray(cn.asarray(distances))
    indices = np.asarray(cn.asarray(indices))

    np.testing.assert_allclose(distances, ref_distances, rtol=1e-6)
    np.testing.assert_equal(indices, ref_indices)


if __name__ == "__main__":
    import sys

    n_rows = int(sys.argv[1])
    n_features = int(sys.argv[2])
    k = int(sys.argv[3])
    dtype = "float32" if len(sys.argv) < 5 else sys.argv[4]

    test_knn(n_rows, n_features, k, dtype)
