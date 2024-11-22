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

import os
import sys
from math import log2
from typing import Union

from numpy.typing import ArrayLike
from scipy.sparse import coo_array, coo_matrix, csr_array, csr_matrix

from legate_raft import add, bincount, fill, log, subtract, sum_over_axis
from legate_raft.core import LogicalStore, as_array, as_store, convert
from legate_raft.naive_bayes import naive_bayes_predict as nb_predict
from legate_raft.prims import count_features, make_monotonic
from legate_raft.sparse import COOStore, PartitionedCOOStore, SparseArray
from legate_raft.utils import _track_provenance

DEBUG = os.environ.get("DEBUG", "0") == "1"


def T(s):
    return s.transpose((1, 0))


class MultinomialNB:
    """Naive Bayes classifier similar to :py:class:`sklearn.naive_bayes.MultinomialNB`.

    Meant to be used together with the the
    :py:class:`~legate_raft.sklearn.feature_extraction.text.HashingVectorizer` and
    :py:class:`~legate_raft.sklearn.feature_extraction.text.TfidfTransformer`.

    Parameters
    ----------
    alpha : float, default=1.0
        Smoothing parameter.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    @_track_provenance
    def fit(
        self, X: Union[SparseArray, COOStore], y: Union[ArrayLike, LogicalStore]
    ) -> "MultinomialNB":
        """Fit naive bayes classifier.

        Parameters
        ----------
        X : SparseArray or COOStore
            The input for `X` is currently meant to be the result of the
            :py:class:`~legate_raft.TfidfTransformer`.
        y : logical store
            The target values.
        """
        X = COOStore.from_sparse_array(X)
        # y = as_store(y)

        if DEBUG:
            print("IN DEBUG MODE!!!", file=sys.stderr)
            import numpy as np
            from numpy.testing import assert_array_almost_equal_nulp, assert_array_equal
            from sklearn.naive_bayes import MultinomialNB as skNB

            sk_estimator = skNB(fit_prior=False)
            sk_estimator.fit(X.to_sparse_array().todense(), as_array(y))

        Y, self._classes_ = make_monotonic(y)

        self.n_classes_ = self._classes_.shape[0]
        self.n_features_ = X.shape[1]

        self._feature_count_ = count_features(X, y, self.n_classes_)

        smoothed_fc = add(self._feature_count_, self.alpha)
        smoothed_cc = sum_over_axis(smoothed_fc, axis=1)
        smoothed_fc_log = log(smoothed_fc)
        smoothed_cc_log = log(smoothed_cc)
        self._feature_log_prob_ = T(subtract(T(smoothed_fc_log), smoothed_cc_log))

        self._class_count_ = bincount(Y, num_bins=self.n_classes_)
        self._class_log_prior_ = fill(
            self.n_classes_,
            -log(self.n_classes_),
            dtype=self._feature_log_prob_.type,
        )

        if DEBUG:
            assert as_array(self._classes_)[0] == 0
            assert (
                as_array(self._feature_count_).dtype
                == sk_estimator.feature_count_.dtype
            )
            assert_array_almost_equal_nulp(
                as_array(self._feature_count_),
                sk_estimator.feature_count_,
                nulp=128,
            )

            smoothed_fc_np = self.feature_count_ + self.alpha
            smoothed_cc_np = np.sum(smoothed_fc_np, axis=1)
            smoothed_fc_log_np = np.log(smoothed_fc_np)
            smoothed_cc_log_np = np.log(smoothed_cc_np)
            assert smoothed_fc_np.dtype == as_array(smoothed_fc).dtype
            assert smoothed_cc_np.dtype == as_array(smoothed_cc).dtype
            assert_array_equal(smoothed_fc_np, as_array(smoothed_fc))
            assert_array_almost_equal_nulp(
                smoothed_cc_np, as_array(smoothed_cc), nulp=1024
            )
            assert_array_almost_equal_nulp(
                smoothed_fc_log_np, as_array(smoothed_fc_log)
            )
            assert_array_almost_equal_nulp(
                smoothed_cc_log_np, as_array(smoothed_cc_log), nulp=128
            )

        return self

    @_track_provenance
    def predict(
        self, X: Union[SparseArray, COOStore, PartitionedCOOStore], *, batch_size=None
    ) -> LogicalStore:
        """
        Predict classes for `X`.
        """
        # if isinstance(X, (SparseArray, COOStore)):  # >= py310
        if isinstance(X, (csr_array, csr_matrix, coo_array, coo_matrix, COOStore)):
            if batch_size is None:
                batch_size = 28

            scale = round(log2(X.nnz))
            n_partitions = max(1, 2 ** (scale - batch_size))
            rows_per_partition = X.nnz // n_partitions

            # if isinstance(X, SparseArray):  # >= py310
            if isinstance(X, (csr_array, csr_matrix, coo_array, coo_matrix)):
                X = COOStore.from_sparse_array(X).partition_by_tiling(
                    rows_per_partition
                )
            else:
                X = X.partition_by_tiling(rows_per_partition)
        elif batch_size is not None:
            raise ValueError(
                "The batch_size parameter can only be provided for sparse_array inputs."
            )

        return nb_predict(
            convert(self._feature_log_prob_, X.type),
            convert(self._class_log_prior_, X.type),
            self._classes_,
            X,
        )

    def __getattr__(self, name):
        # Expose store-objects as arrays when possible.
        try:
            return as_array(self.__getattribute__(f"_{name}"))
        except AttributeError:
            raise AttributeError(name)

    def __getstate__(self):
        state = {
            key: (
                as_array(self.__dict__[key])
                if key.startswith("_")
                else self.__dict__[key]
            )
            for key in [
                "_feature_log_prob_",
                "_class_log_prior_",
                "_classes_",
                "n_classes_",
                "n_features_",
                "_feature_count_",
            ]
            if key in self.__dict__
        }
        return state

    def __setstate__(self, state):
        import numpy as np

        for key, value in state.items():
            if key.startswith("_"):
                self.__dict__[key] = as_store(np.ascontiguousarray(value))
            else:
                self.__dict__[key] = value
