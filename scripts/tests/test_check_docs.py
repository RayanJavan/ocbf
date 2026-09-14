"""Exercise the publication checker with local and versioned MkDocs URLs."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mkdocs.config import load_config

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("check_docs", ROOT / "scripts/check_docs.py")
CHECK_DOCS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_DOCS)


class SiteLinksTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "docs").mkdir()
        # Reuse the real expression; keep plugin setup out of these small link fixtures.
        expression = next(
            line
            for line in (ROOT / "mkdocs.yml").read_text(encoding="utf-8").splitlines()
            if line.startswith("site_url:")
        )
        (self.root / "mkdocs.yml").write_text(
            f"site_name: URL regression fixture\n{expression}\nplugins: []\n",
            encoding="utf-8",
        )
        self.site = self.root / "output" / "html"
        (self.site / "guide").mkdir(parents=True)
        (self.site / "guide/index.html").write_text('<h1 id="topic">Guide</h1>', encoding="utf-8")
        self.env = self.enterContext(patch.dict(os.environ))
        for name in ("DOCS_SITE_URL", "READTHEDOCS_CANONICAL_URL"):
            self.env.pop(name, None)
        self.enterContext(patch.object(CHECK_DOCS, "ROOT", self.root))

    def page(self, href):
        (self.site / "index.html").write_text(f'<a href="{href}">Guide</a>', encoding="utf-8")

    def test_local_fallback_and_relative_links(self):
        config = load_config(config_file=str(self.root / "mkdocs.yml"))
        self.assertEqual(config["site_url"], "http://127.0.0.1:8000/")
        for href in ("/guide/#topic", "guide/#topic"):
            with self.subTest(href=href):
                self.page(href)
                CHECK_DOCS.check_site_links(self.site)

    def test_hosted_version_paths(self):
        for prefix in ("/en/latest/", "/en/v0.1.0/", "/en/42/"):
            with self.subTest(prefix=prefix):
                self.env["READTHEDOCS_CANONICAL_URL"] = f"https://example.invalid{prefix}"
                self.page(f"{prefix}guide/#topic")
                CHECK_DOCS.check_site_links(self.site)

    def test_generic_override_takes_precedence(self):
        self.env["READTHEDOCS_CANONICAL_URL"] = "https://example.invalid/en/latest/"
        self.env["DOCS_SITE_URL"] = "https://example.invalid/project/"
        self.page("/project/guide/#topic")
        CHECK_DOCS.check_site_links(self.site)

    def test_outside_version_is_rejected(self):
        self.env["READTHEDOCS_CANONICAL_URL"] = "https://example.invalid/en/latest/"
        self.page("/en/stable/guide/#topic")
        with self.assertRaisesRegex(ValueError, "link outside site base"):
            CHECK_DOCS.check_site_links(self.site)

    def test_missing_page_and_anchor_are_rejected(self):
        for href, message in (
            ("missing/", "broken built link"),
            ("guide/#missing", "broken built anchor"),
        ):
            with self.subTest(href=href):
                self.page(href)
                with self.assertRaisesRegex(ValueError, message):
                    CHECK_DOCS.check_site_links(self.site)


class RedirectTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "mkdocs.yml").write_text(
            "site_name: Redirect fixture\n"
            "plugins:\n  - redirects:\n      redirect_maps:\n"
            "        old/page.md: new/index.md\n",
            encoding="utf-8",
        )
        self.site = self.root / "site"
        (self.site / "new").mkdir(parents=True)
        (self.site / "new/index.html").write_text("<h1>New</h1>", encoding="utf-8")
        (self.site / "old/page").mkdir(parents=True)
        self.enterContext(patch.object(CHECK_DOCS, "ROOT", self.root))

    def redirect(self, url):
        (self.site / "old/page/index.html").write_text(
            f'<meta http-equiv="refresh" content="0; url={url}">', encoding="utf-8"
        )

    def test_redirect_reaches_mapped_page(self):
        self.redirect("../../new/")
        CHECK_DOCS.check_redirects(self.site)

    def test_wrong_or_missing_redirect_is_rejected(self):
        self.redirect("../../elsewhere/")
        with self.assertRaisesRegex(ValueError, "does not reach"):
            CHECK_DOCS.check_redirects(self.site)
        (self.site / "old/page/index.html").unlink()
        with self.assertRaisesRegex(ValueError, "missing redirect page"):
            CHECK_DOCS.check_redirects(self.site)


if __name__ == "__main__":
    unittest.main()
