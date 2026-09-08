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
| `oracles` | gtsam, pyagrum, problog | the exact-inference oracle — only `gtsam` is wired up |
| `docs` | mkdocs-material, mkdocstrings, … | building this site |
| `dev` | pytest, hypothesis, matplotlib | running the test suite |

Without `bayes` everything still runs — the pipeline falls back to the label-free triplet
estimates. You lose the pooling, which in a sparse regime is most of the value, so install
it unless you have a reason not to.

!!! tip "`gtsam` is optional, and on Windows it needs the CUDA runtime on the DLL path"

    GTSAM backs the [exact oracle](../explanation/inference.md#the-exact-oracle). Nothing in
    the pipeline calls it, so skipping it loses a check rather than a capability — the suite
    skips those tests and passes without it.

    A CUDA-enabled `gtsam.dll` imports the CUDA runtime by name, and Windows resolves that on
    the *DLL search path* rather than on `PATH`, so a plain `import gtsam` fails with a bare
    "DLL load failed". [`ocbf.backends`][ocbf.backends] is the fix: it registers the toolkit
    directories once, and every consumer imports through it.

    ```powershell
    .venv\Scripts\python -m pip install gtsam
    .venv\Scripts\python -c "from ocbf.backends import report; print(report())"
    ```

    `report()` names the directories it searched, so a failure is a path to fix rather than a
    hex address. Set `OCBF_CUDA_BIN` if your toolkit is somewhere non-standard.

## Verify

```bash
.venv/Scripts/python -m pytest -q
```

Expect **154 passed**. The run takes about three minutes, most of it in the end-to-end
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
