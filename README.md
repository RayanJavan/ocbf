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

- [Overview](docs/overview/workflow.md): the workflow and where the library fits.
- [Getting started](docs/getting-started/index.md): installation and a complete assessment.
- [Concepts](docs/concepts/index.md): library objects, their behavior, and concrete examples.
- [How-to guides](docs/how-to/index.md): evidence, trust, inference, queries, and execution.
- [Capabilities](docs/reference/capabilities.md): supported routes and their boundaries.
- [Architecture](docs/development/architecture.md): module responsibilities and extension seams.
- [Integrations](docs/integrations/index.md): application inputs and the Mammut example.
- [Research](research/README.md): separate dated investigations and proposals.

OCBF preserves evidence gaps and model qualifications. Numerical agreement does not
establish physical factory accuracy. The Mammut example requires verified source and
identity bindings before it can produce a retained-data assessment.

## Develop

```bash
python -m pip install -e ".[dev,docs]"
python -m pytest -q
python scripts/check_docs.py
python -m mkdocs build --strict
python scripts/check_docs.py --site-dir site
python -m mkdocs serve
```

The built site includes the generated API reference. CI runs the library checks and
documentation build separately and uploads the `ocbf-site` artifact. See the
[development guide](docs/development/index.md) for validation and contribution rules.
