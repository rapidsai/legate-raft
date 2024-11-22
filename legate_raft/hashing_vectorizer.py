# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import legate.core.types as types
from legate.core import get_legate_runtime
from legate.core import types as ty
from legate_dataframe import LogicalColumn

from legate_raft.library import user_context as library
from legate_raft.library import user_lib
from legate_raft.sparse import COOStore

legate_runtime = get_legate_runtime()


class HashingVectorizer:
    """
    Convert a collection of texts to a matrix of occurrences, this mirrors
    :py:class:`sklearn.feature_extraction.text.HashingVectorizer`.

    Meant to be used together with the the
    :py:class:`~legate_raft.sklearn.feature_extraction.text.TfidfTransformer`
    and :py:class:`~legate_raft.sklearn.naive_bayes.MultinomialNB`.

    Parameters
    ----------
    n_features : int
        Number of features in the output.
    seed : int

    """

    def __init__(self, n_features: int, *, seed: int = 42):
        self.n_features = n_features
        self.seed = seed

    def fit(self, X=None) -> "HashingVectorizer":
        return self

    def transform(self, column: LogicalColumn) -> COOStore:
        output_store = self._transform(column)
        return output_store

    def fit_transform(self, column: LogicalColumn) -> COOStore:
        return self.fit().transform(column)

    def _transform(self, column: LogicalColumn):
        output_store = COOStore.unbound(
            (column.num_rows(), self.n_features), ty.float32
        )

        hv_task = legate_runtime.create_auto_task(
            library, user_lib.cffi.HASHING_VECTORIZER
        )
        column.add_as_next_task_input(hv_task)

        hv_task.add_output(output_store.data)
        hv_task.add_output(output_store.row)
        hv_task.add_output(output_store.col)

        hv_task.add_scalar_arg(self.n_features, types.int32)
        hv_task.add_scalar_arg(self.seed, types.int32)
        hv_task.execute()

        return output_store
