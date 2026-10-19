"""Tests for the Fabric extensions: language parity, new pages and review dates."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import fabric_ext
from fabric_ext import (
    add_language_counterparts,
    counterpart_of,
    fabric_enabled,
    generate_all,
    is_valid_new_page_path,
    language_pairs,
    propose_new_pages,
    set_front_matter_field,
    stamp_review_dates,
    write_page,
)

PAIRS = [("en", "ar")]


@pytest.fixture
def docs(tmp_path):
    """A small bilingual docs tree; the working directory is its root."""
    pages = {
        "en/how-to/upload.md": "---\nowner: t\nlast_reviewed: 2026-01-01\n---\n\n# Upload\n",
        "ar/how-to/upload.md": "---\nowner: t\nlast_reviewed: 2026-01-01\n---\n\n# رفع\n",
        "en/reference/errors.md": "# Errors\n",
        "en/concepts/model.md": "# Model\n",
        "ar/concepts/model.md": "# النموذج\n",
        ".doc-index/en.index.md": "# index\n",
    }
    for rel, text in pages.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


def _llm_reply(text):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )
    return client


class TestConfig:
    def test_language_pairs_ignores_malformed(self):
        cfg = {"language-pairs": [["en", "ar"], ["x"], ["../a", "b"], "en"]}
        assert language_pairs(cfg) == [("en", "ar")]

    def test_fabric_enabled(self):
        assert not fabric_enabled({})
        assert fabric_enabled({"language-pairs": [["en", "ar"]]})
        assert fabric_enabled({"review-date-field": "last_reviewed"})


class TestParity:
    def test_counterpart_paths(self):
        assert counterpart_of("en/how-to/upload.md", PAIRS) == ("ar/how-to/upload.md", False)
        assert counterpart_of("ar/how-to/upload.md", PAIRS) == ("en/how-to/upload.md", True)
        assert counterpart_of("fr/x.md", PAIRS) == (None, False)
        assert counterpart_of("index.md", PAIRS) == (None, False)

    def test_adds_existing_counterpart_only(self, docs):
        files = add_language_counterparts(["en/how-to/upload.md", "en/reference/errors.md"], PAIRS)
        assert "ar/how-to/upload.md" in files
        assert "ar/reference/errors.md" not in files  # does not invent a missing page

    def test_adds_counterpart_for_new_source_page(self, docs):
        files = add_language_counterparts(["en/how-to/reindex.md"], PAIRS)
        assert files == ["en/how-to/reindex.md", "ar/how-to/reindex.md"]

    def test_no_duplicates(self, docs):
        files = add_language_counterparts(["en/how-to/upload.md", "ar/how-to/upload.md"], PAIRS)
        assert files == ["en/how-to/upload.md", "ar/how-to/upload.md"]


class TestNewPages:
    def test_valid_path_rules(self, docs):
        existing = fabric_ext.list_doc_pages()
        assert ".doc-index/en.index.md" not in existing
        assert is_valid_new_page_path("en/how-to/reindex.md", existing, PAIRS)
        assert not is_valid_new_page_path("en/how-to/upload.md", existing, PAIRS)  # exists
        assert not is_valid_new_page_path("ar/how-to/reindex.md", existing, PAIRS)  # not primary
        assert not is_valid_new_page_path("en/new-folder/x.md", existing, PAIRS)  # unknown folder
        assert not is_valid_new_page_path("en/how-to/../../x.md", existing, PAIRS)
        assert not is_valid_new_page_path("en/how-to/Reindex.md", existing, PAIRS)
        assert not is_valid_new_page_path("en/how-to/x.py", existing, PAIRS)

    def test_disabled_by_default(self, docs):
        assert propose_new_pages("diff", [], cfg={}) == []

    def test_parses_and_filters_proposals(self, docs):
        reply = (
            '{"new_pages": [{"path": "en/how-to/reindex.md", "reason": "new endpoint"},'
            ' {"path": "../evil.md", "reason": "x"}]}'
        )
        cfg = {"allow-new-pages": True, "max-new-pages": 2, "language-pairs": [["en", "ar"]]}
        with (
            patch.object(fabric_ext, "get_client", return_value=_llm_reply(reply)),
            patch.object(fabric_ext, "get_model_name", return_value="m"),
        ):
            assert propose_new_pages("diff", [], cfg=cfg) == ["en/how-to/reindex.md"]

    def test_invalid_json_is_ignored(self, docs):
        cfg = {"allow-new-pages": True}
        with (
            patch.object(fabric_ext, "get_client", return_value=_llm_reply("no idea")),
            patch.object(fabric_ext, "get_model_name", return_value="m"),
        ):
            assert propose_new_pages("diff", [], cfg=cfg) == []


class TestGeneration:
    def test_secondary_page_gets_updated_source_as_reference(self, docs):
        calls = {}

        def fake_parallel(diff, files, **kwargs):
            return [(f, "old", f"updated {f}\n") for f in files]

        def fake_update(diff, path, current, **kwargs):
            calls[path] = kwargs["file_instructions"].get(path, "")
            return "ar updated\n"

        cfg = {"language-pairs": [["en", "ar"]]}
        with (
            patch.object(fabric_ext, "_config", return_value=cfg),
            patch.object(fabric_ext, "generate_updates_parallel", side_effect=fake_parallel),
            patch.object(fabric_ext, "ask_ai_for_updated_content", side_effect=fake_update),
        ):
            result = generate_all("diff", ["en/how-to/upload.md", "ar/how-to/upload.md"])

        assert [r[0] for r in result] == ["en/how-to/upload.md", "ar/how-to/upload.md"]
        assert "updated en/how-to/upload.md" in calls["ar/how-to/upload.md"]

    def test_new_pages_are_generated_with_reference(self, docs):
        seen = []

        def fake_new(diff, path, style, instructions, pr, reference=None):
            seen.append((path, reference))
            return f"# {path}\n"

        cfg = {"language-pairs": [["en", "ar"]]}
        with (
            patch.object(fabric_ext, "_config", return_value=cfg),
            patch.object(fabric_ext, "generate_updates_parallel", return_value=[]),
            patch.object(fabric_ext, "generate_new_page", side_effect=fake_new),
        ):
            result = generate_all("diff", ["en/how-to/reindex.md", "ar/how-to/reindex.md"])

        assert [r[1] for r in result] == ["", ""]  # both are new
        assert seen[0] == ("en/how-to/reindex.md", None)
        source = ("en/how-to/reindex.md", "# en/how-to/reindex.md\n")
        assert seen[1] == ("ar/how-to/reindex.md", source)


class TestReviewDates:
    def test_sets_existing_field_only(self):
        text = "---\nowner: t\nlast_reviewed: 2020-01-01\n---\n\nlast_reviewed: body\n"
        out = set_front_matter_field(text, "last_reviewed", "2026-10-19")
        assert "last_reviewed: 2026-10-19\n---" in out
        assert out.endswith("last_reviewed: body\n")
        assert set_front_matter_field("# no front matter\n", "last_reviewed", "x") == (
            "# no front matter\n"
        )

    def test_stamp_uses_configured_field(self):
        files = [("a.md", "", "---\nlast_reviewed: 2020-01-01\n---\n")]
        cfg = {"review-date-field": "last_reviewed", "review-date-timezone": "Asia/Riyadh"}
        with patch.object(fabric_ext, "today_in", return_value="2026-10-19"):
            out = stamp_review_dates(files, cfg=cfg)
        assert "last_reviewed: 2026-10-19" in out[0][2]
        assert stamp_review_dates(files, cfg={}) == files


class TestWritePage:
    def test_creates_folder_for_new_page(self, docs):
        assert write_page("en/how-to/new/page.md", "# New\n")
        assert (docs / "en/how-to/new/page.md").read_text(encoding="utf-8") == "# New\n"

    def test_rejects_unsafe_paths(self, docs):
        assert not write_page("../outside.md", "x")
        assert not write_page("en/script.py", "x")
