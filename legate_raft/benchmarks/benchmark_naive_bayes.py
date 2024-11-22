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
import gc
import os
import sqlite3
from datetime import datetime
from math import log2
from uuid import uuid4

import click
import pandas as pd
from legate.core import get_legate_runtime
from legate.core import types as ty
from mpi4py import MPI

from legate_raft import TfidfTransformer
from legate_raft.benchmarks.util import (  # get_legate_machine_info,
    get_git_info,
    get_system_info,
    timeit,
)
from legate_raft.random import make_docs, randint
from legate_raft.sklearn.naive_bayes import MultinomialNB

legate_runtime = get_legate_runtime()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


KEYS_TO_SHOW_BY_DEFAULT = [
    "iteration",
    "r_scale",
    "c_scale",
    "nnz",
    "implementation",
    "runtime_make_docs_ns",
    "runtime_fit_ns",
    "runtime_predict_ns",
]


def _with_prefix(prefix, mapping):
    return {f"{prefix}{key}": value for key, value in mapping.items()}


def barrier():
    legate_runtime.issue_execution_fence(block=True)


def _run_benchmark(r_scale, c_scale, random_seed, implementation, batch_size):
    # Run the benchmark.
    with timeit() as runtime_make_docs_ns:
        docs = make_docs(
            r_scale, c_scale, random_seed=random_seed, batch_size=batch_size
        )
        X = TfidfTransformer(norm=None).fit_transform(docs)
        y = randint(
            0,
            10,
            shape=(2**r_scale,),
            random_seed=random_seed,
            dtype=ty.int64,
        )

        # Prepartition X for predict stage
        scale = round(log2(X.nnz))
        n_partitions = max(1, 2 ** (scale - batch_size))
        rows_per_partition = X.nnz // n_partitions
        X_partitioned = X.partition_by_tiling(rows_per_partition)
        barrier()

    with timeit() as runtime_ns:
        if "legate":
            estimator = MultinomialNB()
            with timeit() as runtime_fit_ns:
                estimator.fit(X, y)
                barrier()
            with timeit() as runtime_predict_ns:
                y_hat = estimator.predict(X_partitioned)
                assert y_hat.shape[0] == X.shape[0]  # sanity check
                barrier()
        else:
            raise NotImplementedError(implementation)
        barrier()

    return dict(
        ts=datetime.now(),
        random_seed=random_seed,
        r_scale=r_scale,
        c_scale=c_scale,
        nnz=int(X.nnz),
        batch_size=batch_size,
        runtime_ns=runtime_ns(),
        runtime_fit_ns=runtime_fit_ns(),
        runtime_predict_ns=runtime_predict_ns(),
        runtime_make_docs_ns=runtime_make_docs_ns(),
        implementation=implementation,
    )


@click.command()
@click.argument("scale", type=int, default=35)
@click.option("--c_scale", type=int, default=23)
@click.option("--implementation", type=click.Choice(["legate"]), default="legate")
@click.option("-s", "--random-seed", type=int, default=0)
@click.option("-n", "--num-iterations", type=int, default=1)
@click.option("-a", "--all", "show_all", is_flag=True, default=False)
@click.option(
    "--json",
    is_flag=True,
    default=False,
    help="Print output in JSON format (implies -a/-all).",
)
@click.option("--save/--no-save", default=False)
@click.option("--database", default="benchmarks.db")
def benchmark_naive_bayes(
    scale,
    c_scale,
    implementation,
    random_seed,
    num_iterations,
    show_all,
    json,
    save,
    database,
):
    # Determine r_scale as function of total and c_scale
    r_scale = scale - c_scale
    if r_scale < 1:
        raise ValueError(
            f"The selected scale {scale} is too small "
            f"for the requested c_scale {c_scale}."
        )

    # Record unique session id to simplify grouping
    session_id = uuid4().hex

    # Obtain system and legate machine info. Do this first so that we do not run
    # the full benchmark in case it fails.
    system_info = get_system_info()
    # legate_machine_info = get_legate_machine_info()
    git_info = get_git_info()
    mpi_info = dict(size=size)

    gc.disable()

    results = []
    for iteration in range(num_iterations):
        click.echo(f"Running iteration {iteration+1}/{num_iterations}")
        gc.collect()
        result = _run_benchmark(
            r_scale=r_scale,
            c_scale=c_scale,
            random_seed=random_seed + iteration,
            implementation=implementation,
            batch_size=28,
        )
        result["sid"] = session_id
        result["iteration"] = iteration
        result.update(_with_prefix("system.", system_info))
        # result.update(_with_prefix("legate.", legate_machine_info))
        result.update(_with_prefix("git.", git_info))
        result.update(_with_prefix("mpi.", mpi_info))
        results.append(result)

    gc.enable()

    df = pd.DataFrame(results)
    if json:
        click.echo(df.to_json())
    elif show_all:
        click.echo(df)
    else:
        click.echo(df[KEYS_TO_SHOW_BY_DEFAULT])

    if save and rank == 0:
        with sqlite3.connect(database) as conn:
            df.to_sql("naive_bayes", conn, if_exists="append", index=False)


if __name__ == "__main__":
    if os.environ.get("LEGATE_TEST", "") == "1":
        raise RuntimeError("Do not benchmark with LEGATE_TEST set to 1.")

    benchmark_naive_bayes()
