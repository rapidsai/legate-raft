# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from .library import user_lib


class OpCode(IntEnum):
    ADD = user_lib.cffi.ADD
    ADD_CONSTANT = user_lib.cffi.ADD_CONSTANT
    BINCOUNT = user_lib.cffi.BINCOUNT
    CONVERT = user_lib.cffi.CONVERT
    COUNT_FEATURES = user_lib.cffi.COUNT_FEATURES
    FILL = user_lib.cffi.FILL
    FIND_MAX = user_lib.cffi.FIND_MAX
    INVERT_LABELS = user_lib.cffi.INVERT_LABELS
    LOG = user_lib.cffi.LOG
    MAKE_RMAT = user_lib.cffi.MAKE_RMAT
    MAP_LABELS = user_lib.cffi.MAP_LABELS
    MUL = user_lib.cffi.MUL
    MULTIPLY_BY_CONSTANT = user_lib.cffi.MULTIPLY_BY_CONSTANT
    NAIVE_BAYES = user_lib.cffi.NAIVE_BAYES
    POWER = user_lib.cffi.POWER
    RANGE = user_lib.cffi.RANGE
    SUM_OVER_AXIS = user_lib.cffi.SUM_OVER_AXIS
    UNIFORM_INT = user_lib.cffi.UNIFORM_INT
    UNIQUE = user_lib.cffi.UNIQUE

    # From legate.sparsek
    FAST_IMAGE_RANGE = user_lib.cffi.FAST_IMAGE_RANGE
    BOUNDS_FROM_PARTITIONED_COORDINATES = (
        user_lib.cffi.BOUNDS_FROM_PARTITIONED_COORDINATES
    )
    ZIP_TO_RECT1 = user_lib.cffi.ZIP_TO_RECT1
