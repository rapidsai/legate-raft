/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate.h>

namespace legate_raft {

struct Registry {
  static legate::TaskRegistrar& get_registrar();
};

template <typename T, int ID>
struct Task : public legate::LegateTask<T> {
  using Registrar               = Registry;
  static constexpr auto TASK_ID = legate::LocalTaskID{ID};
};

}  // namespace legate_raft
