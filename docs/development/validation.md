# Validation

Validate scientific meaning, numerical computation, and evidence admission at separate
boundaries. Agreement between engines on a synthetic fixture is numerical evidence.

## Routine checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Optional GTSAM checks run when `.[oracles]` is available. Their skips must not hide failure
of the core finite reference route. Contract imports are checked without optional backends.

### Documentation checks

Install the [documentation environment](documentation.md#documentation-environment) first.
These commands work in PowerShell and POSIX shells from the repository root. Confirm
each command succeeds before continuing; PowerShell does not stop on every native command
failure automatically.

```text
python -m pip check
python -m unittest discover -s scripts/tests -v
python scripts/check_docs.py
python -m mkdocs build --strict
python scripts/check_docs.py --site-dir site
```

The URL regression checks cover local, release, preview, and alternate-host paths.
The source check executes the complete documented examples; the strict MkDocs build
generates the API reference through the configured plugins; the publication check inspects
HTML links, anchors, API pages, search, and LLM output. Read the Docs runs the same source
and publication checks around its native strict MkDocs build, using its own output path.

When changing URL or hosting configuration, also build with a version path. This does not
contact the placeholder hostname.

=== "PowerShell"

    ```powershell
    $hadDocsUrl = Test-Path Env:DOCS_SITE_URL
    $previousDocsUrl = $env:DOCS_SITE_URL
    try {
        $env:DOCS_SITE_URL = "https://example.invalid/en/latest/"
        python -m mkdocs build --strict
        if ($LASTEXITCODE -ne 0) { throw "Version-path build failed." }
        python scripts/check_docs.py --site-dir site
        if ($LASTEXITCODE -ne 0) { throw "Version-path publication checks failed." }
    }
    finally {
        if ($hadDocsUrl) {
            $env:DOCS_SITE_URL = $previousDocsUrl
        }
        else {
            Remove-Item Env:DOCS_SITE_URL -ErrorAction SilentlyContinue
        }
    }
    ```

=== "POSIX shell"

    ```bash
    (
      export DOCS_SITE_URL="https://example.invalid/en/latest/"
      python -m mkdocs build --strict && python scripts/check_docs.py --site-dir site
    )
    ```

## Numerical reference seams

| Boundary | Independent check |
|---|---|
| Evidence | Idempotent retransmission, explicit revision conflicts, replacement/retraction, historical cutoffs. |
| Canonical support | Typed multiplicities, inactive states, forbidden configurations, decoding and reduction mass preservation. |
| Exact inference | Enumeration/reference agreement for marginals, conditionals, constants, joints and normalization failures. |
| Hybrid/sampling | Analytical Gaussian references, corrected proposals, constrained support, same-target query estimates with assessed MCSE. |
| BP/EP utilities | Finite enumeration, analytical moments, quadrature, convergence and improper-state behavior. |
| Queries | Applicability, denominators, pending/unresolved outcomes, interval union, shared-history ranks and ties. |
| Reuse/controls | Fresh-versus-reused answers, revised dependencies, expanded support, cancellation and incomplete output. |
| Interchange | Identity preservation, allowlisted reconstruction, array validation and replay. |

Do not replace independent checks with frozen outputs that merely mirror the implementation.

## Numerical assumptions to retain

Copula utilities use monotone observed-to-latent transforms. Unordered categorical values
have no such transform; a censored floor is a latent interval rather than a point. Quantile
clipping prevents infinite latent coordinates, and rank-correlation estimates require
explicit overlap assumptions. These utilities do not establish the physical quality of
the reports used to configure them.

Tail calculations must avoid subtracting nearly equal Gaussian CDF values. Check truncated
moments against independent quadrature in central, one-sided, and far-tail regimes.
For non-Gaussian EP sites, assess quadrature resolution together with the stopping tolerance;
more iterations cannot remove biased local integration. Degree-scaled damping can stabilize
updates but also slow precision accumulation. Check marginal moment residuals, skipped
cavities, proper precision, and agreement with analytical or quadrature references.

For finite BP, compare normalized marginal beliefs with enumeration or the graph oracle
on the same factors and support. A small message residual is not a bound on posterior
error. For canonical sampling, preserve target constants, proposal corrections, support,
and common-history query evaluation; assess error on the actual reported quantity.

## Executable workloads

| Command | Purpose |
|---|---|
| `python -m examples.fixed_parameters` | Finite assessment, named trust, and deterministic revision/replay. |
| `python -m examples.joint_inference` | Same-target exact/sampling comparison and a constrained count reference. |
| `python -m examples.repeated_execution` | Paired fresh/cold/warm trust and query workloads. |
| `python -m examples.resource_validation` | Revision, hybrid, constrained sampling, resources, and warm-start measurements. |

The repeated-execution protocol records seeds, environment, repetitions, retained byte
budgets, numerical tolerances, and a median/MAD comparison. The supplemental workload
checks correctness and resource behavior without a universal speedup gate.

Generated reports belong under ignored `artifacts` directories. Publish a performance
claim only with its workload, environment, inputs, and reproduction command. Avoid fixed
test totals and universal factory-scale claims.

## Evidence admission

Mammut admission checks establish what retained sources contain and whether historical
bindings are adequate. They do not supply physical ground truth. Numerical fixtures and
real-source records remain explicitly distinguished.
