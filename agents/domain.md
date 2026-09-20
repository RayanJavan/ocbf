# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root: the domain glossary. Read it if it exists.
- **`agents/adr/`**: architecture decision records. Read the ADRs that touch the area you're about to work in.

If any of these don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. `/domain-modeling` (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

This is a single-context repo:

```
/
├── CONTEXT.md          ← domain glossary
├── agents/adr/         ← architecture decision records
└── ocbf/               ← library source
```

ADRs live under `agents/`, not the mattpocock default `docs/adr/`: `docs/` is the MkDocs site, built with `--strict` and exact nav-parity (see `scripts/check_docs.py`), so a page there that isn't in the nav fails the build. Keeping convention docs outside `docs/` leaves the published site clean.

## Use the glossary's vocabulary

When your output names a domain concept (an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007, but worth reopening because…_
