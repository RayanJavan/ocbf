# Installation

OCBF requires **Python 3.12 or later**. The commands below install the current checkout.

## Create an environment

=== "PowerShell"

    ```powershell
    git clone https://github.com/RayanJavan/ocbf.git
    cd ocbf
    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -e .
    ```

=== "Linux / macOS"

    ```bash
    git clone https://github.com/RayanJavan/ocbf.git
    cd ocbf
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e .
    ```

If PowerShell does not permit activation, use `.venv\Scripts\python.exe` in place of
`python`. All subsequent commands assume the repository root and selected environment.

## Verify a useful calculation

```bash
python -m examples.fixed_parameters
```

This uses `reference_elimination`, which runs on the core dependencies. It calculates
synthetic query results and verifies replay. Continue with the [quickstart](quickstart.md).

## Install only the extras you need

| Installation | Purpose |
|---|---|
| `pip install -e .` | NumPy/SciPy computation, evidence, models, queries, neutral interchange, and supporting utilities. |
| `pip install -e ".[oracles]"` | Optional GTSAM backend and discrete numerical oracle. |
| `pip install -e ".[dev]"` | Tests and numerical evaluation tooling. |
| `pip install -e ".[docs]"` | MkDocs Material, generated reference, and machine-readable documentation. |

The tutorial selects its engine explicitly. `InferencePolicy()` defaults to
`gtsam_exact`; a bare inference call therefore requires that backend. Automatic routing
is [opt-in](../how-to/choose-inference.md).

### Optional GTSAM on Windows

```bash
python -m pip install -e ".[oracles]"
python -c "from ocbf.backends import report; print(report())"
```

A CUDA-enabled GTSAM build may need its CUDA runtime DLL directory registered.
`ocbf.backends` performs discovery and reports the paths it inspected. Use
`OCBF_CUDA_BIN` when the runtime is installed in a location that discovery does not cover.
This is a backend-loading requirement, not a GPU capability of OCBF.

## Build the documentation

```bash
python -m pip install -e ".[docs]"
python -m mkdocs serve
```

The API reference is generated at build time from the documented public modules.
Use the [development checks](../development/validation.md) when changing the library.
