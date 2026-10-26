"""Tests for [generate-docs]: first draft of a page from chosen code files."""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import fabric_ext
import generate_docs
from generate_docs import (
    GenerateError,
    build_comment,
    doc_type_of,
    is_generate_command,
    load_template,
    parse_model_reply,
    parse_request,
    read_code_files,
    run_generate,
    validate_page_path,
)

PAIRS = [("en", "ar")]

COMMENT = """[generate-docs]
page: en/how-to/manage-collections.md
code: app/main.py, app/models.py
notes: cover create and list only
"""


@pytest.fixture
def repo(tmp_path):
    """A small repo: code, templates and a bilingual docs folder. CWD is the repo root."""
    files = {
        "app/main.py": "def create_collection():\n    '''Create a collection.'''\n",
        "app/models.py": "class Collection: ...\n",
        ".env": "SECRET=1\n",
        ".code-to-docs/templates/how-to.md": "---\nowner: <owning-team>\n---\n\n# <Verb + object>\n",
        "docs/en/how-to/search.md": "# Search\n",
        "docs/ar/how-to/search.md": "# البحث\n",
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


def _reply(text):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )
    return client


class TestRequest:
    def test_detects_command(self):
        assert is_generate_command("[Generate-Docs]\npage: x")
        assert not is_generate_command("[update-docs]")

    def test_parses_fields(self):
        req = parse_request(COMMENT)
        assert req["page"] == "en/how-to/manage-collections.md"
        assert req["code"] == ["app/main.py", "app/models.py"]
        assert req["notes"] == "cover create and list only"
        assert req["overwrite"] is False

    def test_requires_page_and_code(self):
        with pytest.raises(GenerateError):
            parse_request("[generate-docs]\ncode: a.py")
        with pytest.raises(GenerateError):
            parse_request("[generate-docs]\npage: en/how-to/x.md")

    def test_limits_number_of_files(self):
        many = ", ".join(f"f{i}.py" for i in range(11))
        with pytest.raises(GenerateError):
            parse_request(f"[generate-docs]\npage: en/x/y.md\ncode: {many}")

    def test_page_path_rules(self):
        validate_page_path("en/how-to/manage-collections.md", PAIRS)
        for bad in ("../x.md", "/en/x.md", "en/how-to/X.md", "en/how-to/x.py", "ar/how-to/x.md"):
            with pytest.raises(GenerateError):
                validate_page_path(bad, PAIRS)

    def test_doc_type_from_folder(self):
        assert doc_type_of("en/how-to/x.md", PAIRS) == "how-to"
        assert doc_type_of("en/user/how-to/x.md", PAIRS) == "user"
        assert doc_type_of("how-to/x.md", []) == "how-to"


class TestInputs:
    def test_reads_code(self, repo):
        files = read_code_files(["app/main.py", "app/models.py"], repo)
        assert [f for f, _ in files] == ["app/main.py", "app/models.py"]

    def test_refuses_secrets_and_escapes(self, repo):
        for bad in (".env", "../outside.py", "app/missing.py", "config/server.key"):
            with pytest.raises(GenerateError):
                read_code_files([bad], repo)

    def test_size_limit(self, repo):
        (repo / "app/big.py").write_text("x" * (generate_docs.MAX_FILE_CHARS + 1))
        with pytest.raises(GenerateError):
            read_code_files(["app/big.py"], repo)

    def test_template_lookup(self, repo):
        cfg = {"page-templates": {"how-to": ".code-to-docs/templates/how-to.md"}}
        path, text = load_template("how-to", repo, cfg)
        assert path.endswith("how-to.md") and "<owning-team>" in text
        assert load_template("reference", repo, cfg) == ("", "")
        with pytest.raises(GenerateError):
            load_template("how-to", repo, {"page-templates": {"how-to": "missing.md"}})


class TestParity:
    def test_scope(self):
        assert fabric_ext.in_parity_scope("en/user/x.md", ["user/"])
        assert not fabric_ext.in_parity_scope("en/developer/x.md", ["user/"])
        assert fabric_ext.in_parity_scope("en/developer/x.md", None)

    def test_new_counterpart_only_inside_scope(self, repo):
        os.chdir(repo / "docs")
        files = fabric_ext.add_language_counterparts(
            ["en/developer/new.md", "en/user/new.md"], PAIRS, ["user/"]
        )
        assert "ar/user/new.md" in files
        assert "ar/developer/new.md" not in files

    def test_existing_counterpart_kept_outside_scope(self, repo):
        os.chdir(repo / "docs")
        files = fabric_ext.add_language_counterparts(["en/how-to/search.md"], PAIRS, ["user/"])
        assert "ar/how-to/search.md" in files


class TestModelReply:
    def test_parses_json(self):
        page, verify = parse_model_reply(
            json.dumps({"page": "# Title\n\nText", "verify": ["Hourly limit not enforced"]})
        )
        assert page == "# Title\n\nText\n"
        assert verify == ["Hourly limit not enforced"]

    def test_parses_json_in_fences(self):
        page, _ = parse_model_reply('```json\n{"page": "# T", "verify": []}\n```')
        assert page == "# T\n"

    def test_rejects_bad_replies(self):
        for bad in ("not json", '{"page": ""}', '{"verify": []}'):
            with pytest.raises(GenerateError):
                parse_model_reply(bad)


class TestComment:
    def test_lists_files_and_checks(self):
        body = build_comment(
            [("en/how-to/x.md", "# X\n")], ["Owner is a placeholder"], "t.md", ["app/main.py"]
        )
        assert "first draft written from code" in body
        assert "- [ ] Owner is a placeholder" in body
        assert "`en/how-to/x.md`" in body

    def test_error_comment_has_usage(self):
        body = build_comment([], [], "", [], error="Missing `page:` line.")
        assert "Could not generate" in body and "[generate-docs]" in body


class TestRun:
    def _env(self):
        return {"PR_NUMBER": "12", "DOCS_SUBFOLDER": "docs"}

    def _setup_docs(self):
        os.chdir("docs")
        return True

    def test_writes_page_and_counterpart(self, repo):
        cfg = {"language-pairs": [["en", "ar"]], "review-date-field": "last_reviewed",
               "page-templates": {"how-to": ".code-to-docs/templates/how-to.md"}}
        page = "---\nowner: <owning-team>\nlast_reviewed: 2020-01-01\n---\n\n# Manage collections\n"
        reply = json.dumps({"page": page, "verify": ["Confirm the owner"]})
        pushed = {}
        with (
            patch.dict(os.environ, self._env()),
            patch.object(generate_docs, "load_repo_config", return_value=cfg),
            patch.object(generate_docs, "get_client", return_value=_reply(reply)),
            patch.object(generate_docs, "get_model_name", return_value="m"),
            patch.object(generate_docs, "generate_new_page", return_value="---\nlast_reviewed: 2020-01-01\n---\n\n# إدارة المجموعات\n"),
            patch.object(fabric_ext, "today_in", return_value="2026-10-26"),
            patch.object(generate_docs, "commit_and_push", side_effect=lambda d, p, b: pushed.update(paths=p, branch=b)),
            patch.object(generate_docs, "post_comment") as post,
        ):
            run_generate(COMMENT, "", self._setup_docs, lambda n: ("exp-6", "url", False))

        en = (repo / "docs/en/how-to/manage-collections.md").read_text(encoding="utf-8")
        ar = (repo / "docs/ar/how-to/manage-collections.md").read_text(encoding="utf-8")
        assert "last_reviewed: 2026-10-26" in en and "last_reviewed: 2026-10-26" in ar
        assert pushed == {"paths": ["en/how-to/manage-collections.md", "ar/how-to/manage-collections.md"], "branch": "exp-6"}
        assert "Confirm the owner" in post.call_args[0][1]

    def test_refuses_existing_page_without_overwrite(self, repo):
        comment = "[generate-docs]\npage: en/how-to/search.md\ncode: app/main.py\n"
        with (
            patch.dict(os.environ, self._env()),
            patch.object(generate_docs, "load_repo_config", return_value={"language-pairs": [["en", "ar"]]}),
            patch.object(generate_docs, "post_comment") as post,
            pytest.raises(SystemExit),
        ):
            run_generate(comment, "", self._setup_docs, lambda n: ("b", "u", False))
        assert "already exists" in post.call_args[0][1]

    def test_model_error_fails_the_job(self, repo):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("401 invalid key")
        with (
            patch.dict(os.environ, self._env()),
            patch.object(generate_docs, "load_repo_config", return_value={}),
            patch.object(generate_docs, "get_client", return_value=client),
            patch.object(generate_docs, "get_model_name", return_value="m"),
            patch.object(generate_docs, "post_comment") as post,
            pytest.raises(SystemExit),
        ):
            run_generate(COMMENT, "", self._setup_docs, lambda n: ("b", "u", False))
        assert "401 invalid key" in post.call_args[0][1]
        assert not (repo / "docs/en/how-to/manage-collections.md").exists()
