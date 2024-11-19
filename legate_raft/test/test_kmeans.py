# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

import cupynumeric as cn
import numpy as np
import pytest
from legate.core import get_legate_runtime
from numpy.testing import assert_allclose, assert_array_equal
from sklearn import cluster as sk_cluster

from legate_raft.random import make_blobs
from legate_raft.sklearn.cluster import KMeans

legate_runtime = get_legate_runtime()


@pytest.mark.parametrize(
    "n_rows, n_cols, k, dtype",
    [
        (20000, 10, 12, "float32"),
        (50000, 33, 8, "float64"),
        (200, 2, 1, "float64"),
    ],
)
def test_kmeans_training(n_rows, n_cols, k, dtype, verbose=False):
    # Note that this test is currently random, although it should not unless
    # convergence is difficult.
    # Increasing the number of rows stabilizes the test.

    # TODO: make_blobs doesn't support float64 currently (I think)
    #       This is the only reason for cupynumeric (below use can be as_array).
    X, y = make_blobs(n_rows, n_cols, k)
    X = cn.asarray(X)
    X = X.astype(dtype, copy=False)

    model = KMeans(k)
    model.fit(X)

    # Sort centers by the first dimension for comparison:
    lg_sorter = model.cluster_centers_[:, 0].argsort()
    lg_centers = model.cluster_centers_[lg_sorter]
    assert lg_centers.shape == (k, n_cols)

    X_arr = np.asarray(X)
    sklearn_model = sk_cluster.KMeans(k, n_init="auto")
    sklearn_model.fit(X_arr)

    sk_sorter = sklearn_model.cluster_centers_[:, 0].argsort()
    sk_centers = sklearn_model.cluster_centers_[sk_sorter]

    # TODO: Split into it's own test
    # Before using the model, let's try pickling and loading it:
    model = pickle.loads(pickle.dumps(model))

    if verbose:
        print(f"Centers after {model.n_iter_} iterations are:")
        print(repr(lg_centers))
        print(f"SKlearn centers after {model.n_iter_} iterations are:")
        print(repr(sk_centers))

    # Check that the sklearn result and our result are within tolerance
    # TODO: Should probably check the relative difference of the norm.
    assert_allclose(lg_centers, sk_centers, rtol=1e-4)

    # Also compare the found labels (sklearn already adds the attribute)
    sk_labels = sklearn_model.labels_
    lg_labels = np.asarray(cn.asarray(model.predict(X)))

    # Sort labels the same way we sorted the clusters. This is the reverse
    # of the sorter, one way is to argsort again:
    sk_labels = np.argsort(sk_sorter)[sk_labels]
    lg_labels = np.argsort(lg_sorter)[lg_labels]

    assert_array_equal(sk_labels, lg_labels)


if __name__ == "__main__":
    import sys

    n_rows = int(sys.argv[1])
    n_cols = int(sys.argv[2])
    k = int(sys.argv[3])
    dtype = "float32" if len(sys.argv) < 5 else sys.argv[4]

    print(f"Manually testing kmeans training: {n_rows} {n_cols} {k}")
    test_kmeans_training(n_rows, n_cols, k, dtype, verbose=True)
