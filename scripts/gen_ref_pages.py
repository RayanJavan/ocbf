"""Emit one API-reference stub per module, and the nav that ties them together.

Run by ``mkdocs-gen-files`` at build time, so the reference cannot drift out of sync with
the package: a new module appears in the nav the moment it is written, and a deleted one
stops appearing. Nothing produced here is written to disk -- ``mkdocs_gen_files.open``
puts the files into MkDocs' virtual file system for that one build.

Two conventions are load-bearing:

* A package's ``__init__`` becomes the section's ``index.md`` rather than a leaf page, which
  is what Material's ``navigation.indexes`` feature binds to. The ``mkdocs-section-index``
  plugin does the same job and conflicts with it -- do not add both.
* The nav is handed to ``mkdocs-literate-nav`` as ``reference/SUMMARY.md``. That is why
  ``mkdocs.yml`` says ``- API reference: reference/`` with a trailing slash; the slash is
  what tells literate-nav to look for a summary file there.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

# The layout is flat -- the package sits beside pyproject.toml rather than under src/ --
# so paths are taken relative to the repository root and the walk is scoped to the package.
# Walking the root instead would sweep in tests/ and this script itself.
ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "ocbf"

nav = mkdocs_gen_files.Nav()

for path in sorted(PACKAGE.rglob("*.py")):
    module_path = path.relative_to(ROOT).with_suffix("")
    doc_path = path.relative_to(ROOT).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {'.'.join(parts)}\n")

    # Makes "edit this page" on a generated stub point at the source module it documents,
    # rather than at a file that exists only during the build.
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(ROOT))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
