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
from legate.core import types as ty

from legate_raft.sparse.api import COOStore
from legate_raft.utils import _track_provenance

from .make_rmat import make_rmat
from .rng import randint


@_track_provenance
def make_docs(
    n_samples_scale,
    n_features_scale,
    random_seed=0,
    *,
    density=0.001,
    a=0.5,
    b=0.1,
    c=0.1,
    max_term_count=int(1e6),
    batch_size=None,
):
    shape, row, col = make_rmat(
        r_scale=n_samples_scale,
        c_scale=n_features_scale,
        random_seed=random_seed,
        density=density,
        a=a,
        b=b,
        c=c,
        batch_size=batch_size,
    )

    nnz = row.shape[0]
    data = randint(0, max_term_count, (nnz,), ty.int32, random_seed=random_seed)

    return COOStore(
        data=data,
        row=row,
        col=col,
        shape=shape,
    )
