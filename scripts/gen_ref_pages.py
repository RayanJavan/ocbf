"""Generate grouped source references from the explicit public inventory."""

from pathlib import Path
import runpy

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parent.parent
inventory = runpy.run_path(str(ROOT / "scripts/reference_inventory.py"))
nav = mkdocs_gen_files.Nav()
nav[("Overview",)] = "index.md"

with mkdocs_gen_files.open("reference/api/index.md", "w") as page:
    page.write(
        "# API reference\n\n"
        "Start with **Workflow** for composition, then use the scientific contract groups. "
        "**Numerical utilities** and **Source diagnostics and evaluation** expose narrower "
        "interfaces for independent numerical work.\n\n"
        "Signatures and behavior come from the documented source. "
        "Every public member shows its fully typed, cross-referenced signature; a "
        "docstring, where present, adds prose. "
        "[Capabilities](../capabilities.md) describe supported combinations.\n"
    )

for group, module, source in inventory["validate_inventory"](ROOT):
    relative = Path(*module.split(".")).with_suffix(".md")
    destination = Path("reference/api") / relative
    nav[(group, module)] = relative.as_posix()
    with mkdocs_gen_files.open(destination, "w") as page:
        page.write(f"# {module}\n\n::: {module}\n")
    mkdocs_gen_files.set_edit_path(destination, "../" + source.relative_to(ROOT).as_posix())

with mkdocs_gen_files.open("reference/api/SUMMARY.md", "w") as page:
    page.writelines(nav.build_literate_nav())
