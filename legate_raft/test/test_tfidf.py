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

import legate.core.types as ty
from numpy.testing import assert_array_almost_equal_nulp
from sklearn.feature_extraction.text import TfidfTransformer as skTfidfTransformer

from legate_raft.random import make_docs
from legate_raft.sklearn.feature_extraction.text import TfidfTransformer


def test_tfidf_transform():
    docs = make_docs(12, 10)

    sk_transformer = skTfidfTransformer(norm=None)
    X_transformed_sk = sk_transformer.fit_transform(docs.to_sparse_array())
    # Ensure float32 to compare both sensibly below:
    X_transformed_sk = X_transformed_sk.astype("float32", copy=False)

    legate_transformer = TfidfTransformer(norm=None, output_type=ty.float32)
    X_transformed_lg = legate_transformer.fit_transform(docs)

    assert docs.nnz == X_transformed_sk.nnz
    assert docs.nnz == X_transformed_lg.nnz

    assert_array_almost_equal_nulp(
        X_transformed_sk.todense(),
        X_transformed_lg.to_sparse_array().todense(),
        nulp=2,
    )
