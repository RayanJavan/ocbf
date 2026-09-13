# Extension contracts

Pass explicit named/versioned implementations through local registries. There is no
process-global discovery hook or universal plugin base class.

| Extension | Contract and responsibilities |
|---|---|
| Interpreter | `EvidenceInterpreter`: `interpret(record, context)` returns observations and admission issues. |
| Observation channel | `ObservationChannel` supplies finite factors; `ContributingChannel` can also declare shared variables through `ChannelContribution`. |
| Factor kernel | Finite `log_value`, optional batched `log_values`, state `log_density`, or admitted `gaussian_terms`. |
| Engine | `InferenceEngine.assess` evaluates compatibility/cost; `solve` returns a neutral result. |
| Proposal | `ProposalKernel` returns sampled state and forward/reverse log probabilities or densities. |
| Evaluator | `QueryEvaluator.requirements` declares scope/capabilities; `evaluate` returns a qualified estimate. |
| Joint evaluator | `JointQueryEvaluator.evaluate_on` consumes shared evaluation data. |
| Store/control | Runtime ports enable optional bounded reuse and cooperative execution. |

## Keep semantic responsibilities local

The interpreter determines report meaning and scope. The channel owns the observation law.
A factor kernel must preserve declared support and normalization. The engine must not
reinterpret evidence or fit parameters. An evaluator must not fetch evidence or launch
unrecorded inference.

Unknown fields remain admission issues. Model semantics, including priors and dependence,
belong in versioned values rather than mutable implementation closures.

## Supply registries

Use the facade's `interpreters`, `channels`, `factors`, `engines`, and `evaluators`
arguments at their respective boundaries. Built-ins are used when the optional numerical
registries are omitted. Registrations are explicit replacements/compositions, not hidden
mutations of a global registry.

Custom blocked proposals are passed to `BlockedEngine(proposals={block: proposal})`.
Blocks use execution-codec keys and must cover the sampled scope; the complete proposed
state and proposal correction are validated.

## Add optional execution support

Controlled engines expose `assess_with_context` and `solve_with_context`.
Controlled evaluators/posteriors supply their corresponding context-aware operations.
Requesting unsupported controls fails explicitly.

Reusable interpreters, channels, and kernels can declare an immutable `reuse_key`
covering implementation version and every configuration dependency. Missing dependencies
disable numerical reuse. Unknown support compatibility cannot authorize a warm start.

Large finite kernels should implement batched `log_values`. Scalar-only custom lowering
has a bounded fallback; implementing a protocol does not prove tractability.

## Validate and replay

Add independent small reference checks, unsupported-capability checks, and applicable
reuse/invalidation checks. Keep inputs/results neutral and preserve units and origins.

A replay requiring external implementations must receive those registries again.
Named extensions are not automatically portable executable code. See
[validation](validation.md) and [export/replay](../how-to/export-replay.md).
