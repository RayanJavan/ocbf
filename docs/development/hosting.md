# Documentation hosting

Read the Docs Community can host OCBF's public documentation. The repository supplies
the build recipe; Read the Docs manages publication, previews, and version routing.
Adding `.readthedocs.yaml` does not itself create or connect a hosted project.

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

## Connect Read the Docs Community

1. Merge the setup PR after GitHub's **Documentation build** and **Test suite (Python
   3.12)** checks pass. The imported default branch must contain `.readthedocs.yaml` and
   `requirements/docs.txt`.
2. Confirm that the repository is public and has your intended open-source license.
   [Community hosting](https://docs.readthedocs.com/platform/stable/choosing-a-site.html)
   is intended for public open-source documentation.
3. Sign in to [Read the Docs Community](https://app.readthedocs.org/). Install the
   [Community GitHub App](https://github.com/apps/read-the-docs-community/installations/new/)
   under `RayanJavan` and grant it access to `ocbf`. Community and Business have separate
   accounts and Apps. The App manages events and preview integration; a repository PAT or
   manual webhook is not part of this build setup.
4. Choose **Projects → Add project**, select `RayanJavan/ocbf`, and import it. If the
   repository is missing, check the App's repository access and refresh the repository
   list. Set the default branch to `main`, configuration file to `.readthedocs.yaml`,
   language to English, and initial default version to `latest`. Keep versioned hosting.
5. Inspect the first build log. Confirm pinned installation, source/example checks,
   strict MkDocs generation, and publication checks all succeed. Open **View docs** and
   review the pages, search, API reference, diagrams, and source-edit links.
6. Copy the actual hosted URL from **View docs** into the README and GitHub's
   **About → Website** field. The project slug is assigned during import; do not assume
   that `ocbf.readthedocs.io` is available.

See the upstream [project import guide](https://docs.readthedocs.com/platform/stable/intro/add-project.html)
and [Git integration reference](https://docs.readthedocs.com/platform/stable/reference/git-integration.html)
for dashboard details.

## Previews and required checks

In Read the Docs, enable **Settings → Pull request builds → Build pull requests for this
project**. Open a documentation PR and verify a successful status and working preview.
Push a new commit if the PR predates enabling previews. Treat Community previews as public.

Once the preview status has appeared, protect `main` with required PRs and successful
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
