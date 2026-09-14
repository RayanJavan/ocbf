"""Check documentation ownership, executable examples and the publication boundary."""

import argparse
import ast
import json
import re
import runpy
import subprocess
import sys
import tempfile
import textwrap
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import yaml
from mkdocs.config import load_config

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
STALE = re.compile(
    r"\b(?:stage[ -]?[1-4]\b|legacy\b|design\s+(?:doc|record)\b|ocbf\.pipeline|"
    r"ocbf\.sources\.legacy|examples\.stage[34])",
    re.IGNORECASE,
)
LOCAL = re.compile(r"[A-Za-z]:[/\\]Users[/\\]|/(?:home|Users)/[^/\s]+/")
FENCE = re.compile(r"^[ \t]*\x60{3}python[ \t]*\n(.*?)^[ \t]*\x60{3}[ \t]*$", re.M | re.S)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nav_pages(value):
    if isinstance(value, str):
        if value.endswith(".md"):
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from nav_pages(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from nav_pages(item)


def python_blocks(path):
    """Return complete Python blocks; named source excerpts are verified by MkDocs."""
    return [
        textwrap.dedent(block).strip()
        for block in FENCE.findall(path.read_text(encoding="utf-8"))
        if "--8<--" not in block
    ]


def check_sources():
    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    pages = set(nav_pages(config["nav"]))
    authored = {p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.md")}
    require(
        authored == pages,
        f"navigation mismatch: unlisted={authored - pages}, absent={pages - authored}",
    )
    require(not (DOCS / "research").exists(), "research must be outside the published docs tree")
    for path in [
        ROOT / "README.md",
        *DOCS.rglob("*.md"),
        *(ROOT / "ocbf").rglob("*.py"),
        *(ROOT / "examples").rglob("*.py"),
    ]:
        source = path.read_text(encoding="utf-8")
        require(not STALE.search(source), f"obsolete narrative/API in {path.relative_to(ROOT)}")
        require(not LOCAL.search(source), f"developer-local path in {path.relative_to(ROOT)}")
        require(
            not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", source),
            f"unexpected control character in {path.relative_to(ROOT)}",
        )
    for path in DOCS.rglob("*.md"):
        for block in python_blocks(path):
            ast.parse(block, filename=str(path))
    # README links are not checked by the MkDocs build.
    readme = ROOT / "README.md"
    for target in re.findall(r"\]\(([^)]+)\)", readme.read_text(encoding="utf-8")):
        url = urlsplit(target)
        if not url.scheme and url.path:
            require((ROOT / unquote(url.path)).exists(), f"README link does not exist: {target}")
    inventory = runpy.run_path(str(ROOT / "scripts/reference_inventory.py"))
    modules = list(inventory["validate_inventory"](ROOT))
    facade = ast.parse((ROOT / "ocbf/api.py").read_text(encoding="utf-8"))
    for node in facade.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            require(
                bool(ast.get_docstring(node)), f"facade function lacks documentation: {node.name}"
            )
    print(f"Source checks passed: {len(authored)} pages, {len(modules)} public reference modules.")
    return modules


def check_examples():
    """Run documented programs in a temporary working directory, using the real library."""
    scripts = (
        "getting-started/quickstart.md",
        "how-to/interpret-evidence.md",
        "how-to/revise-evidence.md",
        "how-to/configure-trust.md",
        "how-to/configure-constraints.md",
        "how-to/choose-inference.md",
        "how-to/evaluate-queries.md",
        "how-to/read-diagnostics.md",
        "how-to/repeated-execution.md",
        "reference/configuration.md",
        "concepts/semantics.md",
        "concepts/parameters.md",
        "concepts/inference.md",
    )
    quickstart = python_blocks(DOCS / scripts[0])[0]
    with tempfile.TemporaryDirectory(prefix="ocbf-doc-examples-") as temporary:
        prelude = (
            f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
            "from examples.fixed_parameters import run_example\nrun_example()\n"
        )
        subprocess.run(
            [sys.executable, "-c", prelude],
            cwd=temporary,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        for relative in (*scripts, "how-to/export-replay.md"):
            blocks = python_blocks(DOCS / relative)
            program = "\n\n".join(blocks)
            if relative == "how-to/export-replay.md":
                program = quickstart + "\n\n" + program
            source = f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n" + program
            result = subprocess.run(
                [sys.executable, "-c", source],
                cwd=temporary,
                capture_output=True,
                text=True,
                timeout=240,
            )
            require(
                result.returncode == 0, f"documented program failed: {relative}\n{result.stderr}"
            )
            if relative == scripts[0]:
                require(
                    "duration 0.5 probability 0.6282608695652174" in result.stdout,
                    f"quickstart output drifted from the documented block: {relative}",
                )
            print(f"Example passed: {relative}")


def check_site(directory, modules):
    directory = directory.resolve()
    require((directory / "index.html").is_file(), f"no built site at {directory}")
    forbidden = (
        "ocbf-refactoring-plan",
        "ocbf-software-architecture",
        "ocbf-grounding-synthesis",
        "ocbf-inference-synthesis",
        "research-notes.md",
        "stage-1-2.md",
        "stage-3.md",
        "stage-4.md",
    )
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix in (".html", ".json", ".txt", ".md"):
            content = path.read_text(encoding="utf-8")
            require(not STALE.search(content), f"obsolete content published: {path}")
            require(not LOCAL.search(content), f"local path published: {path}")
            require(
                not any(token in content for token in forbidden), f"research/record leaked: {path}"
            )
        require(
            "research" not in path.relative_to(directory).parts, f"research path published: {path}"
        )
    llms = (directory / "llms.txt").read_text(encoding="utf-8")
    full = (directory / "llms-full.txt").read_text(encoding="utf-8")
    require("reference/api/ocbf/api/" in llms, "facade missing from llms.txt")
    require("prepare_evidence" in full, "facade missing from llms-full.txt")
    search = json.loads((directory / "search/search_index.json").read_text(encoding="utf-8"))
    require(
        any("reference/api/ocbf/api/" in item["location"] for item in search["docs"]),
        "facade missing from search",
    )
    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    edit_base = urljoin(config["repo_url"] + "/", config["edit_uri"])
    for _, module, source in modules:
        target = directory / "reference/api" / Path(*module.split(".")) / "index.html"
        require(target.is_file(), f"missing generated public reference: {module}")
        edit_url = urljoin(edit_base, "../" + source.relative_to(ROOT).as_posix())
        require(
            f'href="{edit_url}"' in target.read_text(encoding="utf-8"),
            f"missing source-edit link: {module}",
        )
    print("Publication checks passed: reference, search and LLM corpus are current.")


class PageLinks(HTMLParser):
    """Collect HTML targets without depending on a browser or theme implementation."""

    def __init__(self, source):
        super().__init__()
        self.anchors = set()
        self.links = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.anchors.add(attrs["id"])
        if tag == "a":
            if attrs.get("name"):
                self.anchors.add(attrs["name"])
            if attrs.get("href"):
                self.links.append(attrs["href"])


def check_site_links(directory):
    directory = directory.resolve()
    # Use MkDocs' resolved configuration, including !ENV, for the URL seen by the build.
    config = load_config(config_file=str(ROOT / "mkdocs.yml"))
    base_path = urlsplit(config["site_url"]).path
    pages = {
        path.resolve(): PageLinks(path.read_text(encoding="utf-8"))
        for path in directory.rglob("*.html")
    }
    checked = set()
    for path, page in pages.items():
        for href in page.links:
            link = urlsplit(href)
            if link.scheme or link.netloc:
                continue
            if link.path.startswith("/"):
                require(link.path.startswith(base_path), f"link outside site base: {href}")
                target = (directory / unquote(link.path[len(base_path) :])).resolve()
            else:
                target = (path.parent / unquote(link.path)).resolve() if link.path else path
            if target.is_dir():
                target /= "index.html"
            key = (target, unquote(link.fragment))
            if key in checked:
                continue
            checked.add(key)
            require(target.is_file(), f"broken built link: {path.relative_to(directory)} -> {href}")
            if link.fragment and target in pages:
                require(
                    key[1] in pages[target].anchors,
                    f"broken built anchor: {path.relative_to(directory)} -> {href}",
                )
    print(f"Built-link checks passed: {len(pages)} HTML pages, {len(checked)} targets.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path)
    parser.add_argument("--skip-examples", action="store_true")
    args = parser.parse_args()
    modules = check_sources()
    if args.site_dir:
        check_site(args.site_dir, modules)
        check_site_links(args.site_dir)
    elif not args.skip_examples:
        check_examples()


if __name__ == "__main__":
    main()
