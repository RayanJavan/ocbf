# Queries and result meaning

A query combines a posterior with a process view and a question. Its result includes
the answer's scope, evaluability, and numerical meaning alongside any value.

## Process projections

An `ExecutionProjection` identifies an execution, its population grouping, and candidate
start/end associations. In the examples, executions are Operations and the grouping is a
Job. A `QuerySpec` carries one question's projections, time window, reference, and any
condition intervals; a `QueryBundle` groups those specifications for evaluation.

The same candidate event can affect more than one possible projection. Selecting only
the most likely association would discard that uncertainty before evaluation.
Joint evaluation retains the alternatives supported by the posterior.

## Operation and Job questions

The synthetic assessment compares possible durations with a supplied 30-minute reference.
It evaluates duration, count, and exposure. Whole-Job conformance and priority distributions
are illustrated by the separate [joint-inference example](../how-to/joint-inference.md).

For the introductory Operation, the short endpoint gives a 15-minute duration and the
long endpoint gives a 60-minute duration. Only the latter violates the 30-minute reference.
The duration-exception answer weighs these possibilities among applicable, closed,
evaluable executions. The expected-count answer also reflects how much of the population
is evaluable, so it need not equal the conditional violation probability even for one
Operation.

In the two-Job example, a shared source mode affects endpoint associations across Jobs.
Ranking both Jobs within the same sampled history preserves that relationship. Ranking
their mean exception counts answers a different question from how often each Job ranks
first across histories.

Exposure has a different unit again: minutes overlapping a supplied condition interval.
The [query capability table](../reference/capabilities.md#process-queries) defines the
complete set of supported query kinds and their requirements.

## Evaluability and missing values

An open execution at the horizon is not a measured zero-minute duration. An unresolved
endpoint binding is not evidence that the duration reference was satisfied.
Results retain pending, unresolved, and inapplicable outcomes rather than treating them
as ordinary non-violations.

In the introductory fixture, a history with the start but no associated end is unresolved:
the elapsed time at the horizon exceeds the reference, but a missing end report does not
establish an observed duration. The precise pending/unresolved rules belong to the
[result reference](../reference/results.md#query-estimates).

An `Estimate` can contain a scalar `value`, a structured `distribution`, or no assessed
scalar value. Its `denominator`, `outcomes`, and `qualifications` explain what contributed.
The [result reference](../reference/results.md) defines the field meanings and status rules.

## Different sources of uncertainty

| Situation | Interpretation |
|---|---|
| Both short and long endpoint histories remain plausible | Posterior uncertainty within the supplied model. |
| Repeated sampled runs give slightly different estimates | Monte Carlo error in estimating that posterior quantity. |
| Nominal and cautious channel settings give different answers | Sensitivity to manual assumptions. |
| The endpoint producer's identity mapping is unverified | An evidence or admission limitation. |

More samples address numerical precision. They do not determine which manual trust
setting is appropriate or verify a producer's identity mapping.

## Lineage and exposure

Evidence identities trace the reports behind an assessment. They do not measure causal
responsibility. Likewise, a ten-minute overlap with a tracking-loss interval describes
coincidence in time; it does not establish ten minutes of delay caused by tracking loss.

The [query guide](../how-to/evaluate-queries.md) covers query construction, while
[diagnostics](../how-to/read-diagnostics.md) covers inspection of computed results.
