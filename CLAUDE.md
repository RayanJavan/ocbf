
## init

### rules

- if an assumption affects a task's scope and is hard to resolve, subjective, or not inferrable from the current state, ensure informing the user about it before touching anything.
- strictly avoid accessing anything in .claudeignore in any way shape or form either explicitly or implicitly.
- clarify the scope of your tasks within the codebase files and architecture before diving into them.
- when writing anything, ensure it satisfies what it serves in that context and location in the intended manner and nothing more, and avoid narration and story telling and scope creep. make sure ephemeral context doesn't pollute lasting writings. asking user for clarification when it's needed is encouraged.
- in any documentation and docstrings, aim for accessibility and coherence and be sensitive to duplication, scope creep, and patterns that make the evolution process diverge and become unviable

## Agent skills

Repo configuration the mattpocock engineering skills read. Edit the `agents/*.md` files to change it.

### Issue tracker

Issues and specs live as GitHub issues, driven through the `gh` CLI. See `agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles map to identically named labels. See `agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `agents/adr/` at the repo root, both outside the MkDocs `docs/` tree. See `agents/domain.md`.
