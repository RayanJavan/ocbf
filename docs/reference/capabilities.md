# Capabilities and limitations

Capabilities apply to the declared target and supported inputs. Model validity, numerical
quality, and physical evidence admission are separate assessments.
The [inference topic](../concepts/inference.md#posterior-representations) explains the
representations below; the [selection guide](../how-to/choose-inference.md) uses them in code.

## Inference routes

| Route | Admitted input | Posterior capability | Restrictions |
|---|---|---|---|
| `reference_elimination` | Finite canonical factors | Marginal, joint, expectation, normalizer, joint draws | Input, clique, and requested-joint budgets; core dependencies. |
| `gtsam_exact` | Finite canonical factors | Same finite capabilities | Optional GTSAM; retained native conditionals checked against the canonical reference. |
| `reference_hybrid` | Bounded finite modes and supported linear Gaussian factors | Normalizer and coherent joint draws | Component/dimension limits; exactness only for admitted Gaussian integration. |
| `blocked` | Supported finite/hybrid targets and proposals | Joint draws with numerical assessment | Explicit sampling configuration/RNG; empirical precision and mode exploration. |
| `bp` | Strictly positive finite factors | Approximate single-variable marginals | Explicit `allow_approximate=True`; rejects structural zeros and joint-history requirements; no cooperative control port. |
| `run_ep_adapter` | Caller-grounded proper Gaussian graph | Qualified Gaussian marginal moments | Separate API, latent units; no canonical hybrid compilation, generic joint histories, or parameter fitting. |

`auto` assesses reference finite elimination, reference hybrid integration, then blocked
sampling. It is opt-in and does not include BP or EP. A named engine retains explicit
failure behavior.

The discrete `exact_marginals` graph oracle is a separate numerical utility, returning
marginals and optional MPE. It does not supply the canonical normalizer/joint contract.

## Observation channels

| Channel | Parameter contract | Boundary |
|---|---|---|
| `binary` | `ChannelValues` sensitivity and false-positive probability | Explicit Boolean report semantics. |
| `categorical` | Hit rate or confusion matrix | Active/inactive structural type domains require explicit confusion. |
| `conjunction` | Binary channel values | One report about a declared conjunction; not independent votes for its components. |
| `association_mode` | `AssociationParameters` | Candidate/report confusion by shared finite source mode, with an explicit mode prior. |
| `timestamp_gaussian` | `TimestampParameters` | Supplied real variable, proper prior, units, bias/error scale, optional shared clock bias, and declared inactivity treatment. |

Known copies share one information contribution. Different information groups require
declared dependence assumptions or explicit shared factors. Unsupported soft reports and
selection mechanisms remain outside unjustified likelihoods.

## Model and factor boundaries

The compiler supports finite variables, tables, built-in structural/count/rule factors,
explicit lifecycle bindings, and bounded continuous Gaussian/order constructions.
It preserves hard support, relevant constants, and decoding metadata.

Typed E2O/O2O multiplicities are classified explicitly. Candidate expansions are caller
inputs. Arbitrary attribute models, universal nonlinear integration, automatic entity
resolution, and a general probabilistic programming language are not provided.

## Process queries

| `QuerySpec.kind` | Quantity | Required interpretation |
|---|---|---|
| `duration_exception` | Conditional duration-violation probability | Applicable, closed, evaluable execution denominator and supplied threshold. |
| `expected_exception_count` | Expected evaluable violation count | Explicit population and retained excluded outcomes. |
| `exposure` | Mean descriptive interval overlap | Verified coverage, joint endpoint associations, half-open interval union. |
| `whole_job_conformance` | Distribution of a declared Job's obligation status | Joint scope; violated/unresolved/pending/satisfied/inapplicable treatment. |
| `priority_distribution` | Rank distribution by evaluable violation count | Common histories, fixed population, competition ties, unrankable mass. |

Built-in process evaluators require suitable joint tables or draws. A marginal-only result
cannot satisfy them merely because each individual assertion has a probability.

## Execution and interchange

Sessions provide bounded in-memory reuse, dependency-aware invalidation, conservative
warm starts, controls, and neutral exports. Reuse can compute afresh when an extension's
dependencies are unknown. Controls require an explicit supported execution port.

Not provided: persistent computational caches, general native incremental solving,
distributed execution, streaming ingestion, predictive dynamics, automatic trust learning
inside canonical inference, causal intervention allocation, or standard OCEL file I/O.

See [configuration](configuration.md) for default budgets and
[results](results.md) for output qualifications.
