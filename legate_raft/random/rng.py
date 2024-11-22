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
from legate.core import get_legate_runtime

from legate_raft.library import user_context as library
from legate_raft.library import user_lib
from legate_raft.utils import _track_provenance

legate_runtime = get_legate_runtime()


@_track_provenance
def randint(low, high=None, shape=None, dtype=ty.int32, *, random_seed=0):
    assert dtype in (ty.int32, ty.int64, ty.uint32, ty.uint64)

    if shape is None:
        shape = (1,)

    if high is None:
        high = low
        low = 0

    output = legate_runtime.create_store(dtype, shape)
    task = legate_runtime.create_auto_task(library, user_lib.cffi.UNIFORM_INT)
    task.add_scalar_arg(random_seed, ty.uint64)
    task.add_scalar_arg(low, dtype)
    task.add_scalar_arg(high, dtype)
    task.add_output(output)
    task.execute()

    return output
