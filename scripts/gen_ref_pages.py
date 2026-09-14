"""Generate grouped source references from the explicit public inventory."""

from pathlib import Path
import runpy

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parent.parent
# Nested navigation lists stay near this size by splitting on package prefixes.
MAX_ITEMS = 7


def place(nav, modules, path, depth=1):
    """Add module pages under path, nesting by package prefix when a list grows too long."""
    if len(modules) <= MAX_ITEMS:
        for module in modules:
            nav[(*path, module)] = Path(*module.split(".")).with_suffix(".md").as_posix()
        return
    buckets = {}
    for module in modules:
        buckets.setdefault(".".join(module.split(".")[: depth + 1]), []).append(module)
    if len(buckets) == 1:
        place(nav, modules, path, depth + 1)
        return
    for prefix, members in buckets.items():
        if len(members) == 1:
            place(nav, members, path, depth + 1)
        else:
            place(nav, members, (*path, prefix), depth + 1)


def build_nav(inventory):
    nav = mkdocs_gen_files.Nav()
    nav[("Overview",)] = "index.md"
    grouped = {}
    for group, module, source in inventory["validate_inventory"](ROOT):
        grouped.setdefault(group, []).append((module, source))
    for group, entries in grouped.items():
        place(nav, [module for module, _ in entries], (inventory["tier_of"](group), group))
    return nav, grouped


def main():
    inventory = runpy.run_path(str(ROOT / "scripts/reference_inventory.py"))
    nav, grouped = build_nav(inventory)

    with mkdocs_gen_files.open("reference/api/index.md", "w") as page:
        page.write(
            "# API reference\n\n"
            "**Workflow and contracts** starts with the `ocbf.api` facade, then the scientific "
            "contracts in the order a calculation uses them. **Numerical and source utilities** "
            "expose narrower interfaces for independent numerical work; a low-level marginal "
            "interface does not imply a process-query capability.\n\n"
            "Signatures and behavior come from the documented source. "
            "Every public member shows its fully typed, cross-referenced signature; a "
            "docstring, where present, adds prose. "
            "[Capabilities](../capabilities.md) describe supported combinations.\n"
        )

    for entries in grouped.values():
        for module, source in entries:
            destination = Path("reference/api", *module.split(".")).with_suffix(".md")
            with mkdocs_gen_files.open(destination, "w") as page:
                page.write(f"# {module}\n\n::: {module}\n")
            mkdocs_gen_files.set_edit_path(destination, "../" + source.relative_to(ROOT).as_posix())

    with mkdocs_gen_files.open("reference/api/SUMMARY.md", "w") as page:
        page.writelines(nav.build_literate_nav())


main()
