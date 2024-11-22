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
from legate.core import get_legate_runtime
from legate.core import types as ty

from .cffi import OpCode
from .core import LogicalStore
from .library import user_context as library
from .sparse import PartitionedCOOStore

legate_runtime = get_legate_runtime()


def naive_bayes_predict(
    feature_log_prob: LogicalStore,
    class_log_prior: LogicalStore,
    classes: LogicalStore,
    X: PartitionedCOOStore,
) -> LogicalStore:
    m, k = X.shape
    n, k_ = feature_log_prob.shape

    assert feature_log_prob.type == class_log_prior.type
    assert class_log_prior.type == X.data.store().type
    assert classes.type == ty.int64
    assert X.row.store().type == X.col.store().type == ty.uint64
    assert X.data.store().type == feature_log_prob.type

    # Create output store
    result = legate_runtime.create_store(ty.int64, (m,))

    # Determine launch_domain and create task
    task = legate_runtime.create_manual_task(
        library, OpCode.NAIVE_BAYES, X.row.color_shape
    )

    # Add inputs and outputs
    task.add_input(X.row)
    task.add_input(X.col)
    task.add_input(X.data)
    task.add_input(feature_log_prob)
    task.add_input(class_log_prior)
    task.add_input(classes)
    task.add_output(result)

    # Scalars
    task.add_scalar_arg(m, ty.int32)
    task.add_scalar_arg(k, ty.int32)
    task.add_scalar_arg(n, ty.int32)

    task.execute()

    return result
