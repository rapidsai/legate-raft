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
from legate.core import LogicalStore

from .array_api import add, exp, log
from .array_api import max as lg_max
from .array_api import subtract, sum_over_axis


def logsumexp(x: LogicalStore, axis: int) -> LogicalStore:
    # The implementation below implements the following operations
    # expressed via the numpy API:
    # c = x.max()
    # c + np.log(np.sum(np.exp(x - c)))

    x_max = lg_max(x, axis=axis)
    tmp0 = subtract(x.transpose((1, 0)), x_max).transpose((1, 0))
    tmp = exp(tmp0)
    s = sum_over_axis(tmp, axis=axis)
    out = log(s)
    ret = add(out, x_max)
    return ret
