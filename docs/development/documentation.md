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
readable. Hosting remains the CI site artifact; publishing is a separate action.
