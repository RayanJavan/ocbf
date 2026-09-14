# Documentation maintenance

Keep one principal page for each subject and link to it from other paths.

| Content | Owner |
|---|---|
| Purpose and workflow orientation | Overview |
| Installation and first result | Getting started and executable finite example |
| Procedures | How-to guides |
| Core objects, behavior, and result meaning | Concepts |
| Supported combinations and limitations | Reference tables |
| Callable behavior | Source docstrings |
| Module dependencies and extensions | Development |
| Producer-specific prerequisites | Integration documentation |
| Dated investigations | Repository research corpus, outside the site |

Concepts describes the library's main objects and their behavior through concrete examples.
Topic headings name the subject, prose explains what happens, and short examples expose
relevant inputs or outputs. Sequential instructions belong in Getting started or How-to;
derivations and implementation mathematics belong in numerical reference or research.

When a topic changes scope, update its title, filename, navigation entry, incoming link
labels, and machine-readable output together. Concepts illustrates behavior; Reference
owns exhaustive capability and result-field tables. Repeated summaries should link to
that owner rather than become another independently maintained catalog.

## Build responsibilities

| Owner | Responsibility |
|---|---|
| `pyproject.toml`, `docs` extra | Handwritten compatible dependency ranges |
| `requirements/docs.txt` | Generated pins for the Linux/Python 3.12 build environment |
| `mkdocs.yml` | Navigation, Material theme, Markdown extensions, plugins, and site URL |
| Source docstrings and `scripts/reference_inventory.py` | API content and public-module selection |
| `scripts/gen_ref_pages.py` | Generate API pages through MkDocs' `gen-files` plugin |
| `scripts/check_docs.py` | Shared source, executable-example, and publication checks |
| `.github/workflows/ci.yml` | Validation and the downloadable `ocbf-site` artifact |
| `.readthedocs.yaml` | Hosted environment and hooks around the native MkDocs build |
| Read the Docs project settings | Repository connection, previews, versions, and domains |

Keep the configuration in one `mkdocs.yml`. The library does not import the documentation
toolchain, and the checker does not need a hosting account. Hosting details belong in
[documentation hosting](hosting.md).

Reuse the upstream documentation for general features:
[MkDocs configuration](https://www.mkdocs.org/user-guide/configuration/),
[Material authoring components](https://squidfunk.github.io/mkdocs-material/reference/), and
[mkdocstrings Python options](https://mkdocstrings.github.io/python/usage/configuration/).
This page records OCBF's choices and commands.

## Documentation environment

Use Python 3.12 for parity with documentation CI and Read the Docs. From the repository
root, create an environment if you do not already have one:

=== "PowerShell"

    ```powershell
    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

=== "POSIX shell"

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    ```

If PowerShell blocks activation, replace `python` in the following commands with
`.\.venv\Scripts\python.exe`; no execution-policy change is needed.

Install the committed pins and the checkout together, just as CI and Read the Docs do:

```text
python -m pip install -r requirements/docs.txt -e ".[docs]"
python -m pip check
python -m mkdocs serve
```

The last command starts a local preview; stop it with Ctrl+C. Use the
[documentation checks](validation.md#documentation-checks) before committing. Documentation
does not require the `dev` or `oracles` extra. Keep Ruff in the `docs` extra because
mkdocstrings uses it to format signatures consistently.

The pins target Linux/Python 3.12. Windows can consume the same file, but dependencies
selected only on Windows may be resolved additionally by pip; use WSL for exact parity
with the hosted Python dependency set. The pins do not freeze Python patch releases,
build-system dependencies, or remote Python/NumPy inventories.

## Updating dependency pins

Edit compatible ranges in `pyproject.toml` when needed. Generate pins on Linux/Python 3.12
using [pip-tools](https://pip-tools.readthedocs.io/en/stable/), then review the diff and run
the documentation checks. Ordinary builds consume the committed pins without resolving
a new lock file. Keep resolver tooling separate from the documentation environment.

=== "PowerShell with WSL"

    These commands assume WSL has Python 3.12 and its `venv` support installed. Run them
    from the checkout root; `wsl --exec` uses that directory. The Linux environment must
    be separate from the Windows `.venv`.

    ```powershell
    wsl --exec python3.12 -m venv /tmp/ocbf-docs-lock
    wsl --exec /tmp/ocbf-docs-lock/bin/python -m pip install pip-tools==7.6.1
    wsl --exec /tmp/ocbf-docs-lock/bin/python -m piptools compile --extra docs --strip-extras --output-file requirements/docs.txt pyproject.toml
    ```

=== "Linux"

    ```bash
    python3.12 -m venv /tmp/ocbf-docs-lock
    /tmp/ocbf-docs-lock/bin/python -m pip install pip-tools==7.6.1
    /tmp/ocbf-docs-lock/bin/python -m piptools compile --extra docs --strip-extras --output-file requirements/docs.txt pyproject.toml
    ```

Recreate the temporary resolver environment if it is removed. The compile command reuses
existing pins when compatible; add `--upgrade` for an intentional dependency refresh.
Reinstall with the command above, validate, and commit `pyproject.toml` and
`requirements/docs.txt` together when the dependency declarations change.

## When behavior changes

Update the owning docstring/reference, affected guide, and meaningful validation in the
same change. Describe actual parameters, return values, errors, defaults, side effects,
resource ownership, and capability restrictions.

New public modules must be assigned to a reference group. Private helper modules are
excluded. The reference generator validates its explicit module inventory and source paths;
it never imports optional solvers to decide what to publish. Public members render a fully
typed signature even without prose; run `scripts/check_docstrings.py` to list the members
that still lack a docstring.

## Examples and excerpts

Examples run from the checkout root with the documented environment. Identify synthetic
inputs, provide all prerequisites, and keep generic study helpers independent of
integration-specific data access.

Tutorial construction excerpts use named source regions. Interactive examples with complete
setup are exercised by the documentation checks; construction fragments are labeled as
sketches or source excerpts.

## Page and build checks

Every published Markdown page needs an intentional navigation location. Check source
links, anchors, code samples, generated API targets, search, and machine-readable output.

Research must stay outside the docs tree. Removing a page from navigation alone does not
remove it from generated HTML or search. Generated API pages and current guidance are
included in `llms.txt` / `llms-full.txt`.

Use the [validation commands](validation.md#routine-checks). The site-content check also
rejects leaked research, removed APIs, historical stage narratives, and developer-local
paths. A strict MkDocs build is necessary but cannot detect all misleading prose.

## Visual review

Review Home, the tutorial, a guide, and an API page at desktop and narrow widths in both
light and dark themes. Check diagrams, horizontal table/code scrolling, navigation, search,
copy controls, heading hierarchy, and source-edit links.

Use Material's existing components and a small stylesheet. Keep contrast and focus states
readable. The CI artifact is useful for review; Read the Docs hosts the same generated
site with the [hosting configuration and settings](hosting.md).
