# Installation

OCBF requires **Python 3.12 or later**.

## Install

=== "Windows (PowerShell)"

    ```powershell
    git clone https://github.com/RayanJavan/ocbf.git
    cd ocbf
    py -3.12 -m venv .venv
    .venv\Scripts\python -m pip install -e ".[bayes,dev]"
    ```

=== "Linux / macOS"

    ```bash
    git clone https://github.com/RayanJavan/ocbf.git
    cd ocbf
    python3.12 -m venv .venv
    .venv/bin/python -m pip install -e ".[bayes,dev]"
    ```

## Dependency groups

| extra | brings in | needed for |
| --- | --- | --- |
| *(base)* | numpy, scipy, pandas, networkx | the factor graph, BP engine, diagnostics, baselines |
| `bayes` | pymc, arviz | the hierarchical reliability GLM — the parameter block |
| `inference` | torch | reserved for the planned GPU message-passing backend |
| `io` | pm4py | OCEL 2.0 file round-tripping *(not yet wired up)* |
| `oracles` | pyagrum, gtsam, problog | optional exact-inference cross-checks |
| `docs` | mkdocs-material, mkdocstrings, … | building this site |
| `dev` | pytest, hypothesis, matplotlib | running the test suite |

Without `bayes` everything still runs — the pipeline falls back to the label-free triplet
estimates. You lose the pooling, which in a sparse regime is most of the value, so install
it unless you have a reason not to.

!!! warning "`gtsam` has no Windows wheels"

    It is an optional verification backend under `oracles` and nothing depends on it. On
    Windows, install the other oracles individually rather than the whole extra:

    ```powershell
    .venv\Scripts\python -m pip install pyagrum problog
    ```

## Verify

```bash
.venv/Scripts/python -m pytest -q
```

Expect **76 passed**. The run takes roughly two minutes, most of it in the end-to-end
tests in `tests/test_pipeline.py`, which fuse real generated worlds rather than fixtures.

To check the numerics specifically:

```bash
.venv/Scripts/python -m pytest tests/test_exactness.py -q
```

This compares belief propagation against brute-force enumeration, including the cardinality
forward–backward recursion across five `(k, lo, hi)` shapes. It is the load-bearing test:
that recursion replaces a `2^k` factor, and an error in it would be invisible in aggregate
metrics while quietly corrupting every constrained group in the model.

## Building the documentation

```bash
.venv/Scripts/python -m pip install -e ".[docs]"
.venv/Scripts/mkdocs serve
```

The site rebuilds on changes to both `docs/` and `ocbf/`, so editing a docstring updates the
API reference live.

## Next

Continue to the **[Quickstart](quickstart.md)**.
