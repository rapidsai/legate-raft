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

import json
import os
import platform
import socket
from contextlib import contextmanager
from time import perf_counter_ns

import git
import psutil
from legate.core import get_legate_runtime

try:
    from numba import cuda
except ImportError:
    cuda = None


legate_runtime = get_legate_runtime()


@contextmanager
def timeit():
    runtime = None
    start = perf_counter_ns()
    try:
        yield lambda: runtime
    finally:
        runtime = perf_counter_ns() - start


def get_legate_machine_info():
    raise NotImplementedError()
    machine = legate_runtime.machine
    machine_info = {str(kind): machine.count(kind) for kind in machine.kinds}
    machine_info["preferred_kind"] = str(machine.preferred_kind)
    return machine_info


def get_system_info():
    try:
        user = os.getlogin()
    except OSError:
        user = None

    info = {
        "system": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "total_memory": psutil.virtual_memory().total,
        "available_memory": psutil.virtual_memory().available,
        "used_memory": psutil.virtual_memory().used,
        "percentage_used_memory": psutil.virtual_memory().percent,
        "hostname": socket.gethostname(),
        "user": user,
        "environment": json.dumps(dict(os.environ)),
    }

    try:
        if cuda is None:
            raise RuntimeError("Unable to import cuda from numba.")
        info["cuda_device_count"] = len(cuda.gpus)
        info["cuda_device_name"] = cuda.gpus[0].name.decode()
        info["cuda_version"] = ".".join(map(str, cuda.runtime.get_version()))
    except Exception as e:
        print(f"Could not retrieve CUDA information: {e}")
        info["cuda_device_count"] = None
        info["cuda_device_name"] = None
        info["cuda_version"] = None

    return info


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    return {
        "commit": str(repo.commit()),
        "dirty": repo.is_dirty(),
    }
