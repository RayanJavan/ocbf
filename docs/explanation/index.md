# Explanation

Why OCBF is shaped the way it is. These pages are for understanding, not for following —
nothing here is a set of steps.

## Reading order

1. **[The regime](the-regime.md)** — sparse, unreliable, numerous sources, and the four
   theoretical results that follow. Read this first; everything else is a consequence.
2. **[The model](the-model.md)** — the latent world, the factor layers, and why each one
   exists.
3. **[Inference](inference.md)** — the two-block scheme, why belief propagation rather than
   sampling, why the continuous engine is the same algorithm over a different message
   algebra, and how the approximation is checked against an exact one.
4. **[Diagnostics](diagnostics.md)** — why identifiability, decidability and effective
   sample size are outputs rather than logging.

## Background documents

Two long-form documents underpin the above. They are the original research and design record,
kept whole rather than summarised away:

- **[Research notes](research-notes.md)** — the literature review and formal grounding.
  Around 200 references across truth discovery, sparse-crowdsourcing theory, weak
  supervision, statistical relational learning, mixed graphical models, entity resolution,
  and uncertainty in process mining. Section 4 is the load-bearing one.
- **[Design record](design-record.md)** — the committed model and program architecture,
  plus §11, which records every place where building the system revised the design.

## The one-paragraph version

Given the structural skeleton of an OCEL 2.0 world and many weak, sparse, typed sources, the
binding constraint is not the data format but the *evidence*: most assertions carry too
little source evidence for any aggregation rule to decide them, and most sources make too few
claims for their reliability to be estimated individually. Both problems are solved by
sharing — belief flows across assertions along the schema's structure, and statistical
strength flows across sources through a pooled hierarchy. Everything else in the design is
downstream of those two moves.
