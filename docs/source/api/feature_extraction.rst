Feature Extraction
==================

Text Feature Extraction
-----------------------

The following two vectorizers are designed to be used in conjunction via
a pipeline, such as::

    from sklearn.pipeline import Pipeline

    from legate_raft.sklearn.feature_extraction.text import (
        HashingVectorizer, TfidfTransformer)
    from legate_raft.sklearn.naive_bayes import MultinomialNB

    bayesTfIDF = Pipeline(
        [
            ("hv", HashingVectorizer(n_features=2**17)),
            ("tf-idf", TfidfTransformer()),
            ("mnb", MultinomialNB()),
        ]
    )

The following are the first two classes defined:

.. autoclass:: legate_raft.sklearn.feature_extraction.text.HashingVectorizer
    :members: fit, fit_transform, transform
    :undoc-members:
    :member-order: groupwise

.. autoclass:: legate_raft.sklearn.feature_extraction.text.TfidfTransformer
    :members: fit, fit_transform, transform
    :undoc-members:
    :member-order: groupwise
