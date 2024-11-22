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

import os
import sqlite3
from datetime import datetime

import click
import pandas as pd

# from legate_raft.benchmarks.util import get_legate_machine_info
from legate_raft.benchmarks.util import get_system_info, timeit
from legate_raft.random import make_rmat


@click.command()
@click.argument("r_scale", type=int, default=24)
@click.option("--c_scale", type=int, default=12)
@click.option("--r_scale_batch", type=int, default=20)
@click.option("-s", "--random-seed", type=int, default=0)
@click.option("-a", "--all", "show_all", is_flag=True, default=False)
@click.option(
    "--json",
    is_flag=True,
    default=False,
    help="Print output in JSON format (implies -a/-all).",
)
@click.option("--save/--no-save", default=False)
@click.option("--database", default="benchmarks.db")
def benchmark_make_rmat(
    r_scale, c_scale, r_scale_batch, random_seed, json, show_all, save, database
):
    # Obtain system and legate machine info. Do this first so that we do not run
    # the full benchmark in case it fails.
    system_info = get_system_info()
    # legate_machine_info = get_legate_machine_info()

    # Run the benchmark.
    with timeit() as runtime_ns:
        shape, row, col = make_rmat(r_scale, c_scale, r_scale_batch=r_scale_batch)

    # Basic sanity check and obtain nnz
    assert shape == (2**r_scale, 2**c_scale)
    assert row.shape == col.shape
    nnz = row.shape[0]

    # Prepare result
    entry = dict(
        ts=datetime.now(),
        random_seed=random_seed,
        r_scale=r_scale,
        c_scale=c_scale,
        r_scale_batch=r_scale_batch,
        nnz=nnz,
        runtime_ns=runtime_ns(),
    )
    entry.update(system_info)
    # entry.update(legate_machine_info)
    df = pd.DataFrame([entry])
    if json:
        click.echo(df.to_json())
    elif show_all:
        click.echo(df)
    else:
        click.echo(df[["r_scale", "c_scale", "r_scale_batch", "nnz", "runtime_ns"]])

    if save:
        with sqlite3.connect(database) as conn:
            df.to_sql("make_rmat", conn, if_exists="append", index=False)


if __name__ == "__main__":
    if os.environ.get("LEGATE_TEST", "") == "1":
        raise RuntimeError("Do not benchmark with LEGATE_TEST set to 1.")

    benchmark_make_rmat()
