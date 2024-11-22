/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

enum LegateRaftOpCode {
  _OP_CODE_BASE = 0,
  ADD,
  ADD_CONSTANT,
  ARG_MAX,
  BINCOUNT,
  CATEGORIZE,
  CONVERT,
  COPY,
  COUNT_FEATURES,
  DIVIDE_BY_CONSTANT,
  EXP,
  FILL,
  FIND_MAX,
  FUSED_1NN,
  HASHING_VECTORIZER,
  HISTOGRAM,
  INVERT_LABELS,
  LOG,
  MAKE_RMAT,
  MAP_LABELS,
  MATMUL,
  MUL,
  MULTIPLY_BY_CONSTANT,
  NAIVE_BAYES,
  POWER,
  RAFT_KMEANS_FIT,
  RAFT_KMEANS_PREDICT,
  RAFT_KNN,
  RANGE,
  SPARSE_COO_MM,
  SUM_OVER_AXIS,
  UNIFORM_INT,
  UNIQUE,
  // From legate.sparse:
  FAST_IMAGE_RANGE,
  BOUNDS_FROM_PARTITIONED_COORDINATES,
  ZIP_TO_RECT1,
  TFIDF_FIT,
  TFIDF_PREDICT,
  MAKE_BLOBS
};
