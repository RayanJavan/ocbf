# Diagnostics

The three diagnostics are **results, not logging**. Each is a precondition the model would
otherwise violate silently, and with sources this weak a posterior without them is a number
with unstated preconditions.

For what to *do* about a bad reading, see
[Read the diagnostics](../how-to/read-diagnostics.md).

---

## Identifiability

**Question:** is source reliability even estimable from this data?

Skills are identifiable only when the source overlap graph is irreducible and contains an odd
cycle (arXiv:1706.06660); equivalently, when the co-occurrence sampling pattern has no
bipartite connected component (arXiv:1904.11608). Those are the same statement, so the test
is a two-colouring BFS.

Three verdicts:

| verdict | condition | consequence |
| --- | --- | --- |
| `IDENTIFIED` | non-bipartite component, ≥ 3 sources | jointly estimable from co-claims |
| `SIGN_AMBIGUOUS` | bipartite component | estimable only up to a global sign flip |
| `PRIOR_ONLY` | too little overlap | reliability comes from the hierarchy, not the claims |

`SIGN_AMBIGUOUS` is the interesting one. With no odd cycle, nothing in the data distinguishes
"these sources are accurate" from "these sources are systematically inverted" — the two have
identical likelihood. The better-than-chance prior picks a branch. Reporting that is strictly
better than returning a confident inversion.

In practice global identifiability is *almost never* achieved in a sparse deployment. That is
the point of measuring it rather than assuming it.

---

## Decidability

**Question:** can this assertion be decided at all?

From the exact error exponent, deciding to error `ε` needs total Chernoff information above
`log(1/ε)`:

```text
evidence(a) = Σ_{s covers a} C_s        against        log(1/ε)
C_s = −min_t log Σ_y p₀(y)^t p₁(y)^(1−t)
```

A channel at chance returns exactly zero — it carries no information, and any method treating
such a source as a vote is fooling itself.

The scale is worth internalising:

| source quality | Chernoff information | sources needed for ε = 0.1 |
| --- | --- | --- |
| 0.55 / 0.55 | 0.005 nats | ~460 |
| 0.60 / 0.60 | 0.020 nats | ~113 |
| 0.75 / 0.75 | 0.144 nats | ~16 |
| 0.90 / 0.90 | 0.511 nats | ~5 |

With weak sources and thin redundancy, most assertions cannot clear the bar. That is not a
defect of the estimator; it is a statement about the evidence.

### Decidability is a lower bound for the structured model

The bound counts only what *source votes* carry. OCBF additionally propagates belief along
referential integrity, the type gate and cardinality, so it resolves assertions no aggregation
rule could decide from votes alone.

Both halves are measured, and together they are the clearest confirmation that the structural
prior is primary:

- For **vote-only** methods, accuracy on the decidable subset runs **+14 to +25 points** above
  overall — the flag predicts accuracy, as its derivation says it should.
- For the **structured** model, accuracy on the *undecidable* subset still beats weighted
  voting — the flag understates what is resolvable.

---

## Effective sample size

**Question:** how many *independent* votes does this assertion really have?

`k` near-duplicate sources are not `k` votes, and with weak sources over-counting correlated
evidence is the fastest route to confident error. The design-effect correction, per declared
family:

```text
ESS_c  = m_c / (1 + (m_c − 1)·ρ_c)
ESS(a) = Σ_c ESS_c
```

`ρ_c` is the within-family **residual** correlation — agreement between cluster-mates beyond
what their individual accuracies already explain. Under conditional independence two sources
with ±1 accuracies `a_i, a_j` agree at rate `a_i·a_j`; anything above that is shared error,
which is exactly what must not be counted twice.

Reporting `ESS(a)` beside `deg(a)` turns "twenty sources agree" from a reassuring number into
a checkable one.

### Why declared families rather than learned dependency

Learning a dependency graph also needs overlap — the same scarcity that makes reliability hard
to estimate. In a thin claim graph the estimate is under-determined, so provenance the
operator already knows (same vendor, same upstream model, same site) beats it. The trade is
explicit: declared provenance is *trusted*, so sources that secretly share an upstream model
while declaring different families will not be corrected.

---

## Composition

The diagnostics compose, and the composition is what makes them honest. Decidability accepts
the ESS deflation, so a copied cluster cannot certify its own assertions:

```text
evidence(a) ← evidence(a) · ESS(a)/deg(a)
```

On the default benchmark this drops the decidable fraction from 21.8 % to 15.0 %. Without the
correction, twenty correlated sources would supply twenty Chernoff terms while providing only
a few independent votes.

## Full treatment

[Design record §7](design-record.md); the underlying theory is
[research notes §4](research-notes.md).
