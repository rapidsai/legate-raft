# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import numpy as np
from legate_dataframe import LogicalColumn
from scipy.sparse import coo_matrix

from legate_raft.core import as_array
from legate_raft.sklearn.feature_extraction.text import HashingVectorizer


def test_hv():
    n_replication = 8

    # input dataset
    series = cudf.Series(["lorem ipsum", "cogito ergo sum", "alea iacta est"])
    long_series = cudf.concat([series] * n_replication)
    df = cudf.DataFrame({"input_strings": long_series})
    column = LogicalColumn.from_cudf(df._columns[0])

    # reference output (sum of each row must be the number of words).
    ref_output = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    ref_output = np.vstack([ref_output] * n_replication)

    hv = HashingVectorizer(n_features=16)
    output = hv.fit_transform(column)

    data = as_array(output.data)
    row = as_array(output.row)
    col = as_array(output.col)
    output_matrix = coo_matrix((data, (row, col)), shape=output.shape)

    np.testing.assert_allclose(output_matrix.todense().A, ref_output)
