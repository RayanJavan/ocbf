# About

- **[Limitations](limitations.md)** — what is not built, what is approximated, and what is
  assumed. Read before relying on a result.
- **[Glossary](glossary.md)** — the vocabulary this project uses, with pointers into the
  code.

## Provenance

OCBF was built in three stages, each recorded in full:

| stage | output |
| --- | --- |
| Research | [Research notes](../explanation/research-notes.md) — literature and formal grounding, ~200 references |
| Design | [Design record §§1–10](../explanation/design-record.md) — the committed model and architecture |
| Implementation | [Design record §11](../explanation/design-record.md) — the findings that revised the design |

§11 is worth reading on its own. Each entry is a place where building and measuring the system
contradicted the plan, including three outright bugs the design had blessed: incoherent link
priors, hierarchical variances collapsing under MAP, and truncated moments collapsing in the
far tail.

## Testing

```bash
.venv/Scripts/python -m pytest -q          # ~3 minutes
```

| file | covers |
| --- | --- |
| `tests/test_exactness.py` | BP against brute-force enumeration — the load-bearing numerics test |
| `tests/test_continuous.py` | copula transforms, and every EP site against a closed form or a numerical integral |
| `tests/test_core.py` | schema, assertions, universe, sources, claims |
| `tests/test_diagnostics.py` | identifiability, decidability, ESS, moment estimators |
| `tests/test_gtsam_oracle.py` | exact elimination as an optional second oracle |
| `tests/test_pipeline.py` | end-to-end, including the mandatory baseline comparison |

The claims made throughout this documentation are each pinned by a test, so a regression
that invalidated one would fail the suite rather than quietly making the docs wrong.
