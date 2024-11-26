# legate-raft

legate-raft is a legate-enabled version of [`libraft`](https://docs.rapids.ai/api/raft/stable/).
The project is still in an experimental state and mainly contains prototype versions
of naive bayes, knn, and kmeans.

## Installing

Legate-raft is available as an anaconda package and can be installed using:
```bash
conda install -c legate -c rapidsai -c conda-forge legate-raft
```
To include development releases add the `legate/label/experimental` channel.

## Building

The `legate-raft` library depends on [`legate-dataframe`](https://github.com/rapidsai/legate-dataframe),
which in turn depends on `legate` and `cupynumeric`.

In most cases, you should not have to compile these from scratch.  You can
create a local development environment using conda:
```bash
conda env create -f conda/environments/all_cuda-125_arch-x86_64.yaml -n legate-raft-dev
conda activate legate-raft-dev
```
This will install all build and test requirements.

After this, you can build and install `legate-raft` using:
```bash
./build.sh
```
note that the install is currently not an in-place build, so you need to
rebuild to test Python changes.

## Usage

The main exposed algorithm are currently:
* kNN via `from legate_raft.knn import run_knn`
* kmeans via `from legate_raft.sklearn.cluster import KMeans` working similar
  to the sklearn version.
* naive bayes via `from legate_raft.sklearn.naive_bayes import MultinomialNB`

All of these and especially any further functionality should be considered
to be in a prototype or exploratory stage.


## Contributing

Please see our [our guide for contributing to legate-raft](CONTRIBUTING.md).
