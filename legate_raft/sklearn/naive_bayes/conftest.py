# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

try:
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.pipeline import Pipeline
except ImportError:
    HAS_SKLEARN = False
else:
    HAS_SKLEARN = True


@pytest.mark.skipif(not HAS_SKLEARN, reason="test requires scikit-learn")
def _nlp_20news():
    try:
        twenty_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
    except:  # noqa E722
        pytest.xfail(reason="Error fetching 20 newsgroup dataset")

    pipeline = Pipeline(
        [
            ("count_vectorizer", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
        ]
    )

    X = pipeline.fit_transform(twenty_train.data)
    Y = np.array(twenty_train.target)

    return X, Y


@pytest.fixture(scope="module")
def nlp_20news():
    return _nlp_20news()
