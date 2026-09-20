# OCBF

**Object-Centric Belief Fusion** is a Python library for reasoning about uncertain
object-centric histories from heterogeneous evidence.

Supply a fixed schema and candidate universe, interpret versioned reports, resolve
manual trust, and compile one probability model. Inference returns declared posterior
capabilities; process queries turn those capabilities into qualified answers about
duration exceptions, conformance, exposure, and priorities.

```text
Semantic context + interpreted evidence + resolved parameters
                              ↓
                       Canonical model
                              ↓
                    Qualified posterior
                              ↓
                     Process query results
```

## Try it

Python 3.12 or later is required. From a checkout:

```bash
git clone https://github.com/RayanJavan/ocbf.git
cd ocbf
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in a POSIX shell, then run:

```bash
python -m pip install -e .
python -m examples.fixed_parameters
```

The example uses synthetic reports and the core finite reference engine. It writes
readable results, manual-trust comparisons, and reproducible correction/retraction
calculations under `artifacts/fixed-parameters`. No native solver or factory data is needed.

## Learn and use

Read the documentation at **[ocbf.readthedocs.io](https://ocbf.readthedocs.io/)**. The
sources for each section are also readable in this repository:

- [Getting started](docs/getting-started/index.md): installation and a complete assessment.
- [Concepts](docs/concepts/index.md): the workflow, library objects, and their behavior.
- [How-to guides](docs/how-to/index.md): evidence, inference, execution, and integration.
- [Capabilities](docs/reference/capabilities.md): supported routes and their boundaries.
- [Architecture](docs/development/architecture.md): module responsibilities and extension seams.
- [Integrate an application](docs/how-to/integrate-application.md): application inputs and the Mammut example.

OCBF preserves evidence gaps and model qualifications. Numerical agreement does not
establish physical factory accuracy. The Mammut example requires verified source and
identity bindings before it can produce a retained-data assessment.

## Develop

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

For documentation, use the pinned environment and PowerShell or POSIX commands in
[documentation maintenance](docs/development/documentation.md#documentation-environment).
MkDocs, Material, and mkdocstrings build the guides and generated API reference together.
CI checks that site separately from the library suite and uploads the `ocbf-site` artifact.
Read the Docs publishes the same build; the [hosting guide](docs/development/hosting.md)
records its settings. See the [development guide](docs/development/index.md) for
validation and contribution rules.
