# Inference and posteriors

Inference computes a posterior for one compiled model. The posterior describes uncertainty
over the model's possible histories. Its representation determines which questions can
be answered from the result.

## Marginals and joint histories

A marginal concerns one variable, such as whether an end event belongs to an Operation.
A joint describes related variables together.

As a separate illustration, consider two endpoint associations with a probability of one
half each. Those marginal values alone do not say whether the associations always occur
together, are independent,
or are mutually exclusive. A duration or conformance question can distinguish these cases.
The introductory model explicitly makes its two end associations mutually exclusive;
that constraint supplies information beyond their individual marginal probabilities.

OCBF's process queries therefore declare their required joint scopes through
`QueryRequirements`. The requirements concern the information needed by the question,
not just a preferred algorithm.

The introductory model makes this concrete. Requesting the joint over its two end
associations shows the exclusivity that the separate marginals cannot:

```python
from examples.fixed_parameters import example_inputs
from ocbf.api import compile_model, infer
from ocbf.assertions import AssertionRef
from ocbf.belief.posterior import QueryRequirements
from ocbf.inference.contracts import InferencePolicy

spec, queries, settings, records = example_inputs()
short = str(AssertionRef.e2o("short", "subject", "op"))
long = str(AssertionRef.e2o("long", "subject", "op"))
result = infer(
    compile_model(spec),
    requirements=QueryRequirements(scopes=((short, long),), capabilities=("marginal", "joint")),
    policy=InferencePolicy(engine="reference_elimination"),
)
joint = result.posterior.joint((short, long)).probabilities
print("P(short end):", round(float(joint[1].sum()), 4))
print("P(long end):", round(float(joint[:, 1].sum()), 4))
print("P(both ends):", round(float(joint[1, 1]), 4))
```

```text
P(short end): 0.425
P(long end): 0.425
P(both ends): 0.0
```

Each end association is individually plausible, yet the joint probability of both is zero:
the exclusivity constraint lives in their joint, not in either marginal.

## Posterior representations

A finite exact calculation retains enough conditional information to answer supported
joint questions or draw complete assignments. It can resolve the small synthetic endpoint
assessment without sampling error.

A sampled posterior retains a collection of joint assignments. Each assignment represents
one coherent possible history. Related queries can use the same histories, including
shared source modes and endpoint choices.

A bounded hybrid calculation combines finite alternatives with supported continuous
Gaussian quantities. For example, an end association can be uncertain alongside its
timestamp. Supported continuous parts are integrated and reconstructed jointly when
histories are drawn.

Approximate marginal adapters have a narrower role. BP provides admitted finite marginal
tables; the separate caller-grounded EP interface provides Gaussian marginal moments.
Those summaries do not automatically supply complete process histories.

The [capability reference](../reference/capabilities.md#inference-routes) defines the exact
engine restrictions and supported operations.

## Plans, policies, and results

| Object | Meaning |
|---|---|
| `QueryRequirements` | The scopes and posterior operations needed by a query bundle. |
| `InferencePolicy` | Engine selection, numerical settings, and planning budgets. |
| Execution plan | The admitted route, its representation, and assessed costs. |
| `InferenceResult` | The posterior, capabilities, identities, diagnostics, and computation qualifications. |

The policy default is `gtsam_exact`. The quickstart explicitly selects
`reference_elimination` so it needs only core dependencies. Automatic routing is opt-in;
it assesses supported routes without changing evidence or fitting missing parameters.

## Sampling behavior

Sampling separates warmup from retained draws and uses explicit random streams.
Increasing draws may improve precision, but a chain that rarely moves between plausible
histories can remain misleadingly concentrated. Numerical assessments describe the
reported quantity and its sampled behavior.

Two sample runs can produce slightly different estimates for the same model. Two exact
runs on the same supported target should agree within numerical precision.
Neither agreement nor convergence establishes the physical accuracy of the source reports.

The [joint-inference guide](../how-to/joint-inference.md) contains executable examples.
