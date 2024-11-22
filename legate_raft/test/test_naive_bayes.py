# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from time import perf_counter

import joblib
import pytest
from legate.core import types as ty
from numpy.testing import (
    assert_array_almost_equal_nulp,
    assert_array_equal,
    assert_equal,
)
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB as skNB

from legate_raft import TfidfTransformer
from legate_raft.core import as_array
from legate_raft.random import make_docs, randint
from legate_raft.sklearn.naive_bayes import MultinomialNB


@pytest.mark.xfail(reason="Test doesn't prepare data as logical stores")
def test_multinomial_nlp20news(nlp_20news):
    X, y = nlp_20news

    tic = perf_counter()
    sk_estimator = skNB(fit_prior=False)  # TODO: consider to use True here
    sk_estimator.fit(X, y)
    sk_y_hat = sk_estimator.predict(X)
    toc = perf_counter()
    print("sklearn", toc - tic)

    tic = perf_counter()
    estimator = MultinomialNB()
    y_hat = estimator.fit(X, y).predict(X)
    toc = perf_counter()
    print("legate", toc - tic)
    y_hat = as_array(y_hat)

    assert_array_almost_equal_nulp(
        estimator.feature_count_, sk_estimator.feature_count_, nulp=20
    )
    # The log_prob values are numerically divergent.
    assert_array_almost_equal_nulp(
        estimator.feature_log_prob_, sk_estimator.feature_log_prob_, nulp=256
    )
    assert_equal(estimator.n_classes_, len(sk_estimator.classes_))
    assert_equal(estimator.class_log_prior_, sk_estimator.class_log_prior_)
    assert_array_equal(estimator.class_count_, sk_estimator.class_count_)
    assert_array_equal(y_hat, sk_y_hat)
    print(accuracy_score(y, sk_y_hat))
    print(accuracy_score(y, y_hat))
    assert accuracy_score(y, y_hat) > 0.9


@pytest.mark.xfail(reason="Test doesn't prepare data as logical stores")
def test_multinomial_save_load(nlp_20news, tmp_path):
    X, y = nlp_20news

    estimator = MultinomialNB()
    estimator.fit(X, y)
    joblib.dump(estimator, tmp_path / "model.pickle")

    y_hat = estimator.predict(X.copy())

    estimator_loaded = joblib.load(tmp_path / "model.pickle")
    y_hat_cmp = estimator_loaded.predict(X.copy())

    assert_array_equal(as_array(y_hat), as_array(y_hat_cmp))


def test_multinomial_with_datagen(verbose=False):
    random_seed = 0
    n_cats = 10
    r_scale = 8
    c_scale = 15
    docs = make_docs(r_scale, c_scale, random_seed=random_seed)
    X = TfidfTransformer(norm=None).fit_transform(docs)
    y = randint(0, n_cats, shape=(2**r_scale,), random_seed=random_seed, dtype=ty.int64)

    X_np = X.to_sparse_array()
    y_np = as_array(y)

    tic = perf_counter()
    sk_estimator = skNB(fit_prior=False)  # TODO: consider to use True here
    sk_y_hat = sk_estimator.fit(X_np, y_np).predict(X_np)
    toc = perf_counter()
    if verbose:
        print("sklearn", toc - tic)

    tic = perf_counter()
    estimator = MultinomialNB()
    y_hat = estimator.fit(X, y).predict(X)
    toc = perf_counter()
    if verbose:
        print("legate", toc - tic)
    y_hat = as_array(y_hat)

    assert_array_almost_equal_nulp(
        estimator.feature_count_, sk_estimator.feature_count_, nulp=20
    )
    # The log_prob values are numerically divergent.
    assert_array_almost_equal_nulp(
        estimator.feature_log_prob_, sk_estimator.feature_log_prob_, nulp=256
    )
    assert_equal(estimator.n_classes_, len(sk_estimator.classes_))
    assert_array_almost_equal_nulp(
        estimator.class_log_prior_, sk_estimator.class_log_prior_, nulp=2
    )
    assert_array_equal(estimator.class_count_, sk_estimator.class_count_)
    assert_array_equal(y_hat, sk_y_hat)
    if verbose:
        print(accuracy_score(y_np, sk_y_hat))
        print(accuracy_score(y_np, y_hat))
        print(accuracy_score(y_np, y_hat))
    assert accuracy_score(y_np, y_hat) > 0.9


if __name__ == "__main__":
    test_multinomial_with_datagen(verbose=True)
