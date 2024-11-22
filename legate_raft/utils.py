# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools

from legate.core import track_provenance as _track_provenance_legate


def _track_provenance(func):
    """
    Private decorator to add "provenance" tracking to all Python side
    functions which end up calling legate tasks.
    All calls which directly launch tasks should be decorated.

    This e.g. adds Python line number to profiling results.  Similar to
    cunumeric, we use `functools.update_wrapper` which the legate core
    version did not at the time of writing.
    """
    wrapped_func = _track_provenance_legate()(func)
    functools.update_wrapper(wrapped_func, func)
    return wrapped_func
