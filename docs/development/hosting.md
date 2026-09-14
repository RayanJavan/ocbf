# Documentation hosting

Read the Docs Community hosts OCBF's public documentation at
[ocbf.readthedocs.io](https://ocbf.readthedocs.io/). The repository supplies the build
recipe; Read the Docs manages publication, previews, and version routing.

## How the build is shared

The [documentation environment](documentation.md#documentation-environment) and
[validation commands](validation.md#documentation-checks) are the same locally, in GitHub
CI, and on Read the Docs. The hosting adapter installs the pins and checkout, checks the
sources and examples, lets Read the Docs run MkDocs in strict mode, then checks the
published files at `$READTHEDOCS_OUTPUT/html`. That variable uses Linux shell syntax in
`.readthedocs.yaml`, even when a contributor works in PowerShell.

Material, mkdocstrings, `gen-files`, literate navigation, search, and the LLM corpus all
remain configured in `mkdocs.yml`. Generated API pages are built from source docstrings
through `scripts/gen_ref_pages.py`; do not run that generator separately. Generated
`site/` and `_readthedocs/` output stays out of Git.

The library test job remains independent. Each service keeps its short orchestration
recipe and passes the appropriate output path to the shared checker.

## URL configuration

MkDocs resolves `site_url` using its native `!ENV` support, in this order:

| Value | Use |
|---|---|
| `DOCS_SITE_URL`, when set | Explicit override for another host or a local integration check |
| `READTHEDOCS_CANONICAL_URL`, when set | URL supplied by Read the Docs for the built version |
| `http://127.0.0.1:8000/` | Local preview fallback |

Leave `DOCS_SITE_URL` unset in Read the Docs so its own version-aware URL takes effect.
The publication checker loads the resolved MkDocs configuration too. Test URL changes
with the [version-path check](validation.md#documentation-checks).

## Project settings

The project is connected through the
[Read the Docs Community GitHub App](https://github.com/apps/read-the-docs-community),
which delivers push and pull request events; no repository token or manual webhook is used.

| Setting | Value |
|---|---|
| Project slug | `ocbf` |
| Repository and default branch | `RayanJavan/ocbf`, `main` |
| Configuration file | `.readthedocs.yaml` at the repository root |
| Language | English |
| Versioning | Multiple versions; default version `latest` |
| Pull request previews | Public |
| GitHub **About → Website** | `ocbf.readthedocs.io` |

Check a changed setting against the first build it affects: pinned installation,
source/example checks, strict MkDocs generation, and publication checks must all succeed.
See the upstream [Git integration reference](https://docs.readthedocs.com/platform/stable/reference/git-integration.html)
for dashboard details.

## Previews and required checks

Pull request builds are controlled by **Settings → Pull request builds → Build pull
requests for this project**. A PR opened before previews were enabled needs a new commit
to receive one. Treat Community previews as public.

Once the preview status has appeared on a PR, protect `main` with required PRs and successful
checks: **Test suite (Python 3.12)**, **Documentation build**, and the exact Read the Docs
status name observed on that PR. A sole maintainer can require PRs and checks without
requiring another person's approval. Keep docs CI enabled for code and example changes
as well as authored documentation changes.

Keep Material's built-in search and Read the Docs' standard version flyout initially.
Material's search index is already checked by OCBF's publication script; Read the Docs'
server-side search does not index PR previews. Custom theme/search integration is optional.
See [PR previews](https://docs.readthedocs.com/platform/stable/guides/pull-requests.html)
and [Material integration](https://docs.readthedocs.com/platform/stable/intro/mkdocs.html).

## Version policy

| Event or version | Intended result |
|---|---|
| Pull request | Temporary preview of that change |
| Merge to `main` | Update `latest` |
| Release tag such as `v0.1.0` | Build from that tagged revision |
| `stable` | Highest qualifying non-prerelease version |

Under Read the Docs **Automation Rules**, add a rule for **Tag**, **SemVer versions**,
**Activate version**. Activate existing inactive versions manually when needed. Retain
`latest` as the default until an intended release builds successfully, then select
`stable` as the default landing version.

Tag only tested commits containing the hosting configuration and dependency pins. Files
added to `main` do not appear in older tags. Read the Docs owns version hosting; a second
deployment workflow or version manager is not needed for this policy.

See [versions](https://docs.readthedocs.com/platform/stable/versions.html),
[automation rules](https://docs.readthedocs.com/platform/stable/automation-rules.html), and
the [configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html).
