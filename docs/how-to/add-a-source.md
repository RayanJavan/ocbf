# Add your own source

A source is anything that emits claims about assertions. Implementing one means supplying a
[`SourceProfile`][ocbf.sources.base.SourceProfile] and a list of
[`Claim`][ocbf.sources.claims.Claim] objects.

## Minimal source

```python
from ocbf.assertions import AssertionRef
from ocbf.sources import (
    ChannelFamily, Claim, CoverageSemantics, SourceProfile, StaticSource,
)

profile = SourceProfile(
    source_id="scanner_07",
    coverage=CoverageSemantics.OPPORTUNISTIC,
    channel=ChannelFamily.BINARY,
    cluster_id="acme_scanners_v3",
    features=(1.0, 250.0),
    feature_names=("is_optical", "sampling_hz"),
)

claims = [
    Claim("scanner_07", AssertionRef.e2o("ev_0012", "item", "item_3"), True),
    Claim("scanner_07", AssertionRef.e2o("ev_0012", "item", "item_9"), False),
]

source = StaticSource(profile, claims)
```

Pass a list of these to [`fuse`][ocbf.pipeline.fuse] and you are done.

## Choose the coverage semantics

This is the field with **no default**, because guessing wrong biases every downstream
number. It declares what the source's *silence* means.

=== "`OPPORTUNISTIC`"

    Silence is uninformative. The source reports on whatever it happened to see; not
    mentioning an assertion says nothing about it.

    ```python
    coverage=CoverageSemantics.OPPORTUNISTIC
    # scope defaults to the claimed refs
    source = StaticSource(profile, claims)
    ```

    Use for: log scrapes, extractions from documents that may or may not mention a fact,
    anything where absence is an artefact of what the source happened to look at.

=== "`COMPLETE_OVER_SCOPE`"

    Silence inside the declared scope is a **negative claim**. The source examined
    everything in scope and reported only what it found.

    ```python
    coverage=CoverageSemantics.COMPLETE_OVER_SCOPE
    source = StaticSource(profile, claims, scope=every_ref_it_examined)
    ```

    Use for: detectors, sensors with a defined field of view, any pipeline that swept a
    fixed candidate set. This is where a detector's false-negative rate does its work —
    treating its silence as "no information" discards the strongest signal it carries.

=== "`SELECTIVE`"

    Silence is informative through a propensity that depends on the truth. The general
    case; the other two are its limits.

    ```python
    coverage=CoverageSemantics.SELECTIVE
    source = StaticSource(profile, claims, scope=candidate_refs)
    ```

    Use for: sources more likely to report on assertions that are true — a human who logs
    exceptions, an alerting system that fires on positives.

!!! warning "A non-opportunistic source must declare a scope wider than its claims"

    `StaticSource` raises if the scope equals the claims, because then there is no silence
    to interpret and the declaration is meaningless. Either widen the scope or use
    `OPPORTUNISTIC`.

## Set `cluster_id` honestly

`cluster_id` is the **declared source family**, and it is the dependency correction — not
metadata. Sources sharing a family share a random effect in the reliability GLM, so their
agreement is partly explained by that shared effect rather than counted as independent
confirmation.

Group by whatever makes two sources *correlated in their errors*:

- same vendor or hardware model
- same upstream ML model or extractor version
- same physical site, shift, or operator
- same parent pipeline

Getting this wrong in the permissive direction (splitting sources that really are
correlated) makes the model overconfident. Twenty sources sharing an upstream model are not
twenty votes.

## Pick a channel family

| `ChannelFamily` | `Claim.value` | Use for |
| --- | --- | --- |
| `BINARY` | `bool` | existence, E2O/O2O links |
| `CATEGORICAL` | `str` | event type, from the assertion's domain |
| `CONTINUOUS` | `float` | timestamps, numeric attributes |
| `DISTRIBUTIONAL` | `soft=` mapping | a learned detector's output distribution |

[`ChannelFamily.for_family`][ocbf.sources.base.ChannelFamily.for_family] gives the natural
default for an assertion family.

For a detector that outputs probabilities, pass them rather than thresholding:

```python
Claim("classifier_2", AssertionRef.event_type("ev_0012"),
      value="Ship", soft={"Ship": 0.7, "Deliver": 0.25, "PackItem": 0.05})
```

## Supply features for pooling

`features` feeds the regression term of the reliability GLM. It is what lets a source with
three claims inherit an accuracy estimate from *comparable* sources rather than from its own
useless sample — and what lets the model say something about a source it has never seen.

Use observable, stable properties: modality, vendor, model version, sampling rate, sensor
placement, latency. Keep the vector the same shape across all sources.

## A streaming or lazy source

`StaticSource` is a convenience. Anything satisfying the
[`Source`][ocbf.sources.base.Source] protocol works:

```python
class DatabaseSource:
    def __init__(self, profile, connection):
        self.profile = profile
        self._conn = connection

    def scope(self):
        return [AssertionRef.parse(row.ref) for row in self._conn.scope_rows()]

    def claims(self):
        for row in self._conn.claim_rows():
            yield Claim(self.profile.source_id, AssertionRef.parse(row.ref), row.value)
```

## Verify before fusing

```python
from ocbf.sources import ClaimSet
from ocbf.diagnostics import overlap_report

claims = ClaimSet.from_sources(my_sources)
print(claims.summary())              # check density, deg_a, deg_s
print(overlap_report(claims).explain())
```

If `deg_a_median` is 1 and the overlap graph is mostly `prior_only`, the model will lean
almost entirely on the structural prior. That may be fine — see
[Read the diagnostics](read-diagnostics.md) — but you should know it going in.

## See also

- [Coverage semantics, derived](../explanation/the-model.md#sources-and-coverage) — why the
  three modes are one family with a single parameter.
- [`ocbf.sources`][ocbf.sources] — the full API.
