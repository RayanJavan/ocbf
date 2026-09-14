# Inference and posteriors

The [model](model.md) lists the possible histories and how each observation weighs them.
**Inference** computes the **posterior**: the probability of each history, given all the evidence
and parameters. Different engines store the posterior in different forms, and each form can answer
different questions. This page therefore starts from the question.

## Choose the question first

The duration question has to read the start link and the end links *together*. A result that only
stored a separate probability for each link could not answer it. A calculation therefore starts by
declaring its **requirements**:

- the groups of assertions whose **joint distribution** is needed, meaning the probability of every
  combination of their values;
- the operations the result must support.

```python
from ocbf.api import plan_inference
from ocbf.belief.posterior import QueryRequirements
from ocbf.inference.contracts import ExecutionPlan

requirements: QueryRequirements = QueryRequirements(
    scopes=((links["short"], links["long"]),),  # (1)!
    capabilities=("marginal", "joint"),  # (2)!
)
policy: InferencePolicy = InferencePolicy(engine="reference_elimination")  # (3)!
plan: ExecutionPlan = plan_inference(model, requirements=requirements, policy=policy)
print(plan.engine, plan.capabilities)
```

1.  One group of assertions whose combinations must be available together: the two end links.
2.  The operations the result must provide: probability tables for single assertions (`marginal`)
    and for groups of assertions (`joint`).
3.  An exact engine that needs only OCBF's core dependencies. Without a policy, OCBF uses
    `gtsam_exact`. With `engine="auto"`, OCBF picks the first engine that meets the requirements.

Running the block prints:

```text
reference_elimination ('marginal', 'joint', 'expectation', 'normalizer', 'joint_draws')
```

The **plan** names the engine that will run and the operations its result will support. This engine
supports more than was requested: it can also compute expected values (`expectation`), the total
weight of all histories before normalizing (`normalizer`), and random samples of complete
histories (`joint_draws`). Planning never changes the evidence or fills in missing parameters. On
the queries page, `requirements_for` works out the requirements from the questions themselves.

## Marginals and joint histories

A **marginal** is the probability distribution of one assertion. A **joint** is the distribution of
several assertions together:

```python
from ocbf.api import infer
from ocbf.belief.posterior import InferenceResult, JointTable

result: InferenceResult = infer(model, requirements=requirements, policy=policy)
for event in ("start", "short", "long"):
    linked: JointTable = result.posterior.marginal(links[event])  # (1)!
    print(f"P({event} linked) = {float(linked.probabilities[1]):.4f}")
ends: JointTable = result.posterior.joint((links["short"], links["long"]))  # (2)!
print(f"P(both ends linked) = {float(ends.probabilities[1, 1]):.4f}")
```

1.  A table with one probability for each value of the assertion: index 0 is `False`, index 1 is
    `True`.
2.  A table with one axis per assertion, in the order given, so `[1, 1]` is "both true".

Running the block prints:

```text
P(start linked) = 0.7391
P(short linked) = 0.4250
P(long linked) = 0.4250
P(both ends linked) = 0.0000
```

Each end link has a probability of 0.425, yet the two are never true together. The marginals
alone cannot show that. Two marginals of 0.425 would fit ends that always occur together, ends that
are unrelated, and ends that never occur together, and the duration question has a different answer
in each case. Only the joint shows which of these holds.

!!! note "Keep in mind"

    Two links can each be plausible and still be impossible together. Answer questions about
    histories from joints or from sampled complete histories, never from a list of separate
    probabilities.

## How plausible each history is

The joint over all three links gives the probability of each history class from the
[model page](model.md#possible-histories):

```python
from itertools import product

link_joint: JointTable = result.posterior.joint(
    (links["start"], links["short"], links["long"])
)


def history_class(start: bool, short: bool, long: bool) -> str:
    if short and long:
        return "two ends (impossible)"
    if not start:
        return "start not linked"
    if short:
        return "closed after 15 min"
    return "closed after 60 min" if long else "open at 12:00"


classes: dict[str, float] = {}
for start, short, long in product((False, True), repeat=3):  # (1)!
    label: str = history_class(start, short, long)
    index: tuple[int, int, int] = (int(start), int(short), int(long))
    classes[label] = classes.get(label, 0.0) + float(link_joint.probabilities[index])
for label, probability in classes.items():
    print(f"{label:<22} {probability:.4f}")
```

1.  All eight combinations of the three links, added up into the history classes of the model page.

Running the block prints:

```text
start not linked       0.2609
two ends (impossible)  0.0000
open at 12:00          0.1109
closed after 60 min    0.3141
closed after 15 min    0.3141
```

This is what the duration question will read:

- The two closed histories are equally likely. The two end reports have the same trust values, and
  nothing else favors either one.
- In about 11% of the probability, `op` started but neither end belongs to it.
- In about 26%, the start event is not linked to `op`, for example because the start report was
  false.
- The impossible history has a probability of exactly zero.

## Posterior representations

The result above is exact: it stores probability tables. A sampling engine stores something else,
a large set of complete histories drawn at random from the posterior:

```python
import numpy as np

from ocbf.inference.contracts import SamplingConfig

sampled: InferenceResult = infer(
    model,
    requirements=QueryRequirements(
        scopes=(tuple(links.values()),), capabilities=("joint_draws",)  # (1)!
    ),
    policy=InferencePolicy(
        engine="blocked", sampling=SamplingConfig(chains=2, warmup=200, draws=1000)  # (2)!
    ),
    rng=np.random.default_rng(1),  # (3)!
)
for label, inferred in (("exact", result), ("sampled", sampled)):
    print(f"{label:<8} {inferred.computation:<24} {inferred.capabilities}")
```

1.  **[Joint draws](../reference/glossary.md)**: complete histories drawn at random, with a value
    for every assertion.
2.  A Markov chain Monte Carlo sampler with a deliberately small configuration: 2 chains, each
    discarding 200 initial steps (warmup) and then keeping 1,000 histories.
3.  The random number generator is passed explicitly, so the same seed reproduces the same draws.

Running the block prints:

```text
exact    exact-on-finite-model    ('marginal', 'joint', 'expectation', 'normalizer', 'joint_draws')
sampled  mcmc-on-declared-target  ('joint_draws',)
```

The exact result computes marginals, joints, and expected values without sampling error, and it
can also draw histories. The sampled result offers only its draws. Each draw is one complete
history, so two quantities computed from the same draws, such as a duration and a count, always
describe the same histories. There is, however, no exact probability table to read.

Other engines exist as well. The hybrid engine also handles uncertain numeric values, such as a
timestamp with a normally distributed error. The approximate engines return only marginals, so they
cannot answer questions that need joints. The
[capability reference](../reference/capabilities.md#inference-routes) lists every engine and what
it supports.

## Numerical qualifications

Every result states how its numbers were computed. `computation` names the kind of number, for
example `exact-on-finite-model`. `diagnostics` holds checks specific to the engine, such as sampler
convergence statistics.

Exact runs on the same model agree to floating-point precision. Sampled runs with different seeds
give slightly different estimates. The **[MCSE](../reference/glossary.md)** (Monte Carlo standard
error) of an estimate tells you how large that difference is expected to be;
[Queries and result meaning](queries.md#different-sources-of-uncertainty) shows it for the duration
question.

More draws make a sampled estimate more precise. A sampler that stays stuck near some histories,
for example one that almost always draws `short` as the end, can still report a small MCSE while
missing `long`. Neither agreement between runs nor convergence says anything about whether the
reports themselves are accurate.

## Summary

- Questions about histories need joint probabilities. Requirements declare which ones, and the plan
  names an engine that can provide them.
- The posterior gives each history class a probability: about 31% closed after 15 minutes, 31%
  closed after 60, 11% open at 12:00, and 26% with the start not linked to `op`.
- The posterior's form, exact tables or sampled histories, decides which questions a result can
  answer and whether its numbers carry sampling error.

Next, [Queries and result meaning](queries.md) turns these probabilities into answers about `op`.
For procedures, see [choose inference](../how-to/choose-inference.md) and
[use joint inference](../how-to/joint-inference.md).
