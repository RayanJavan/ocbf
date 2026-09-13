# Integrations

The application supplies structure, reports, interpretation contracts, and business
references. OCBF supplies the reusable evidence/model/inference/query machinery.

| Application responsibility | OCBF boundary |
|---|---|
| Fetch and retain raw records | Pass immutable evidence envelopes with provenance. |
| Derive schema and candidates | Pass OCBF-owned schema/universe values and a semantic context. |
| Understand producer versions and identity | Register explicit interpreters and bindings. |
| Define business population and criteria | Supply query projections and normative references. |
| Schedule, authorize, persist, and display | Invoke the synchronous facade and consume neutral results. |

No database, Kafka, credentials, scheduler, or UI is required by the core contracts.

## Structural inputs

An external derivation tool can construct OCBF semantic values. The Mammut example's
elastocel bridge consumes the existing structure derivation result; generic OCBF modules
do not import elastocel or factory storage.

A missing candidate universe must be supplied explicitly. A source interpreter cannot
repair an absent semantic binding by inventing a likely production identity.

See the [Mammut example](mammut.md) for its actual admission boundary and
[extension contracts](../development/extensions.md) for reusable implementations.
