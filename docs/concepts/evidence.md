# Evidence and observations

An `EvidenceRecord` retains a source report with its identity, producer version, revision,
knowledge time, and provenance. An `Observation` is the interpreted content used by the
model. The original report and its interpretation remain distinct.

## From a report to an observation

In the synthetic assessment, a report says that endpoint event `short` was reported for
Operation `op`.

| Representation | Contents |
|---|---|
| Report payload | The producer's fields, such as an event identifier and a reported Boolean value. |
| Evidence envelope | The source and producer version, stable record identity, revision identity, time, and provenance. |
| Interpreted observation | The assertions concerned, reported value, channel name, and contributing evidence revisions. |
| Admission issue | A recorded reason that some content could not be interpreted or admitted. |

The example interpreter treats the positive report as a claim that the event exists
**and** belongs to the Operation. It does not turn that report into two independent
measurements. A channel later describes how likely this report would be under each
possible truth.

An unrecognized producer version has no registered interpretation. Evidence preparation
retains an admission issue instead of inventing the missing meaning.

## Effective revisions

A report has a stable record identity and a sequence of explicitly connected revisions.

| Available history | Effective content |
|---|---|
| Initial positive report | The positive report contributes. |
| Replacement reporting a negative value | The replacement contributes instead of the initial report. |
| Retraction of that replacement | The report is withdrawn; the earlier positive report is not restored. |
| Identical retransmission | It does not create another independent observation. |

The knowledge cutoff determines which revisions are available to a snapshot.
A correction learned on Tuesday can concern an event that occurred on Monday.
Monday's historical cutoff and Tuesday's corrected assessment therefore use different
evidence states.

## Shared information and coverage

Two transports can carry the same underlying report. Their shared information identity
allows the model to count that contribution once. Two genuinely distinct measurements
can still share an error cause, such as a common source mode; that dependence needs an
explicit observation model or shared factor.

Silence has meaning only when an observation opportunity and reporting mechanism are
established. A missing endpoint report alone does not establish that the Operation never
ended. A repeated stale position report alone does not establish continuous occupancy.

The [interpretation guide](../how-to/interpret-evidence.md) covers interpreter registration.
The [revision guide](../how-to/revise-evidence.md) covers replacement and retraction calls.
