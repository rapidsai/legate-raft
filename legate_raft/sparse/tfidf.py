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

import numpy as np
from legate.core import get_legate_runtime
from legate.core import types as ty

from legate_raft.core import as_store

from ..core import convert
from ..library import user_context as library
from ..library import user_lib
from ..multiarray import bincount
from ..utils import _track_provenance
from .api import COOStore

DEBUG = os.environ.get("DEBUG", "0") == "1"

legate_runtime = get_legate_runtime()


class NotFittedError(RuntimeError):
    pass


class TfidfTransformer:
    """
    Transform a count matrix to normalized representation, see
    :py:class:`sklearn.feature_extraction.text.TfidfTransformer`
    which this mirrors.

    Meant to be used together with the the
    :py:class:`~legate_raft.sklearn.feature_extraction.text.HashingVectorizer` and
    :py:class:`~legate_raft.sklearn.naive_bayes.MultinomialNB`.

    Parameters
    ----------
    norm : None
        *Included to mirror scikit-learn*
    use_idf : True
        *Included to mirror scikit-learn*
    smooth_idf : bool, default=True
        Prevent division by zero (see scikit-learn documentation for details).
    output_type : legate type, default=float64
        The numerical output type.  Defaults to ``float64``.

    """

    def __init__(self, *, norm=None, use_idf=True, smooth_idf=True, output_type=None):
        assert norm is None
        assert use_idf is True

        if output_type is None:
            output_type = ty.float64

        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.output_type = output_type

        self._idf = None

    @_track_provenance
    def fit(self, X, y=None) -> "TfidfTransformer":
        n_samples, n_features = X.shape

        cols = X.col if isinstance(X, COOStore) else X.indices
        cols = convert(cols, ty.uint64)

        self._idf = self._fit(cols, n_samples, n_features)
        return self

    def _fit(self, cols, n_samples, n_features):
        # bincount to compute document frequencies
        df = bincount(cols, num_bins=n_features, output_type=self.output_type)

        tfidf_fit_task = legate_runtime.create_auto_task(
            library, user_lib.cffi.TFIDF_FIT
        )
        tfidf_fit_task.add_input(df)
        tfidf_fit_task.add_output(df)  # modify in place
        tfidf_fit_task.add_scalar_arg(int(bool(self.smooth_idf)), ty.int32)
        tfidf_fit_task.add_scalar_arg(n_samples, ty.uint64)
        tfidf_fit_task.execute()

        return df

    @_track_provenance
    def transform(self, X):
        try:
            idf = self._idf
        except AttributeError:
            raise NotFittedError

        cols = X.col if isinstance(X, COOStore) else X.indices
        cols = convert(cols, ty.uint64)

        return self._transform(X, cols, idf)

    def _transform(self, X, cols, idf):
        # broadcast idf to data partitions and compute tfidf
        # (could be done element-wise, really)
        # NOTE: The configuration is order dependent
        tfidf_predict_task = legate_runtime.create_auto_task(
            library, user_lib.cffi.TFIDF_PREDICT
        )
        assert cols.type == ty.uint64
        X_data = convert(X.data, idf.type)  # may or may not create a copy

        tfidf_predict_task.add_input(cols)
        tfidf_predict_task.add_input(X_data)
        tfidf_predict_task.add_input(idf)
        tfidf_predict_task.add_output(X_data)  # modify in place
        tfidf_predict_task.add_alignment(cols, X_data)
        tfidf_predict_task.add_broadcast(idf)
        tfidf_predict_task.execute()

        X.data = X_data
        return X

    @_track_provenance
    def fit_transform(self, X, y=None):
        n_samples, n_features = X.shape

        cols = X.col if isinstance(X, COOStore) else X.indices
        cols = convert(cols, ty.uint64)

        self._idf = self._fit(cols, n_samples, n_features)
        return self._transform(X, cols, self._idf)

    def __getstate__(self):
        array_if = (
            self._idf.get_physical_store().get_inline_allocation().__array_interface__
        )

        class wrapper:
            pass

        w = wrapper()
        w.__array_interface__ = array_if
        array = np.array(w, copy=False)
        return {"_idf": array}

    def __setstate__(self, d):
        self._idf = as_store(d["_idf"])
