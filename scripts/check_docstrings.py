"""List public reference members that render as bare signatures without a docstring."""

import argparse
import ast
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent.parent


def _public(name):
    return not name.startswith("_")


def undocumented(source):
    """Names of public top-level defs and public methods lacking a docstring."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    missing = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _public(node.name):
            if not ast.get_docstring(node):
                missing.append(node.name)
        elif isinstance(node, ast.ClassDef) and _public(node.name):
            if not ast.get_docstring(node):
                missing.append(node.name)
            for sub in node.body:
                method = isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                if method and _public(sub.name) and not ast.get_docstring(sub):
                    missing.append(f"{node.name}.{sub.name}")
    return missing


def main():
    inventory = runpy.run_path(str(ROOT / "scripts/reference_inventory.py"))
    total = 0
    modules = 0
    for _, module, source in inventory["validate_inventory"](ROOT):
        gaps = undocumented(source)
        if gaps:
            modules += 1
            total += len(gaps)
            print(f"{module}: {', '.join(gaps)}")
    print(f"\nUndocumented public members: {total} across {modules} modules.")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max", type=int, help="fail when the undocumented count exceeds this ratchet"
    )
    args = parser.parse_args()
    # Without --max this is a report and exits 0; with it, the count may only go down.
    total = main()
    if args.max is not None and total > args.max:
        sys.exit(f"Undocumented public members rose to {total}; the ratchet allows {args.max}.")
