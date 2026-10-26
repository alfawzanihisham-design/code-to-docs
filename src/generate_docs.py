"""
[generate-docs]: write a first draft of a documentation page from chosen code files.

Code-to-Docs normally works from a pull request diff, so it can only document what a
pull request changes. [generate-docs] is for the first version of a page about code
that already exists. The reviewer names the page, the code files to read and,
optionally, the template; the tool writes the page (and its counterpart in the
other language), commits it to the pull request and lists what must be verified.

Comment format (one field per line):

    [generate-docs]
    page: en/how-to/manage-collections.md
    code: app/main.py, app/models.py, app/config.py
    template: how-to            (optional, defaults to the page's folder)
    notes: cover create and list only   (optional)
    overwrite: yes              (optional, needed to replace an existing page)

Templates are mapped in `.code-to-docs/config.json`:

    "page-templates": {"how-to": ".code-to-docs/templates/how-to.md", ...}

Failures are reported in a pull request comment and make the job fail.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from config import get_client, get_model_name, load_repo_config
from fabric_ext import (
    _SAFE_NEW_PATH,
    counterpart_of,
    generate_new_page,
    in_parity_scope,
    language_pairs,
    stamp_review_dates,
    write_page,
)
from generation import _SYSTEM_PROMPT, strip_code_fences, validate_format
from security_utils import run_command_safe, sanitize_output, validate_file_path

COMMAND = "[generate-docs]"
MAX_CODE_FILES = 10
MAX_FILE_CHARS = 60000
MAX_TOTAL_CODE_CHARS = 150000
MAX_TEMPLATE_CHARS = 8000

# Files that must never be sent to the model, whatever the reviewer asks for.
_BLOCKED_NAMES = re.compile(
    r"(^|/)(\.env[^/]*|.*\.(pem|key|p12|pfx|jks|crt)|id_rsa[^/]*|secrets?\.[a-z]+|credentials[^/]*)$",
    re.IGNORECASE,
)

USAGE = (
    "Usage:\n\n"
    "```\n"
    "[generate-docs]\n"
    "page: en/how-to/manage-collections.md\n"
    "code: app/main.py, app/models.py\n"
    "template: how-to   (optional)\n"
    "notes: what to cover   (optional)\n"
    "overwrite: yes   (optional, to replace an existing page)\n"
    "```"
)


class GenerateError(Exception):
    """A problem the reviewer can fix; reported in the PR comment."""


# =============================================================================
# REQUEST
# =============================================================================


def is_generate_command(comment_body):
    return COMMAND in (comment_body or "").lower()


def parse_request(comment_body):
    """Parse the comment into a request dict. Raises GenerateError."""
    fields = {}
    for line in (comment_body or "").splitlines():
        m = re.match(r"^\s*(page|code|template|notes|overwrite)\s*:\s*(.*?)\s*$", line, re.I)
        if m:
            fields[m.group(1).lower()] = m.group(2)

    page = fields.get("page", "").strip().strip("`")
    if not page:
        raise GenerateError("Missing `page:` line.")
    code = [c.strip().strip("`") for c in fields.get("code", "").split(",") if c.strip()]
    if not code:
        raise GenerateError("Missing `code:` line with at least one file.")
    if len(code) > MAX_CODE_FILES:
        raise GenerateError(f"Too many code files ({len(code)}). The limit is {MAX_CODE_FILES}.")

    return {
        "page": page,
        "code": code,
        "template": fields.get("template", "").strip().lower(),
        "notes": fields.get("notes", "").strip(),
        "overwrite": fields.get("overwrite", "").strip().lower() in ("yes", "true", "1"),
    }


def validate_page_path(page, pairs):
    """Check the target page path. Raises GenerateError."""
    if ".." in page or page.startswith("/") or not _SAFE_NEW_PATH.match(page):
        raise GenerateError(
            f"`{page}` is not a valid page path. Use lowercase letters, digits, hyphens and "
            "slashes, ending in .md, relative to the docs folder."
        )
    if pairs:
        primary = {p for p, _ in pairs}
        if page.split("/", 1)[0] not in primary:
            raise GenerateError(
                f"`{page}` must start with a source language folder ({', '.join(sorted(primary))}). "
                "The other language is generated from it."
            )


def doc_type_of(page, pairs):
    """Folder that names the page type, e.g. en/how-to/x.md -> how-to."""
    parts = page.split("/")
    if pairs and parts and parts[0] in {lang for pair in pairs for lang in pair}:
        parts = parts[1:]
    return parts[0] if len(parts) > 1 else ""


# =============================================================================
# INPUTS
# =============================================================================


def read_code_files(paths, repo_root):
    """Read the chosen code files from the repository. Raises GenerateError."""
    root = Path(repo_root)
    files, total = [], 0
    for rel in paths:
        if _BLOCKED_NAMES.search(rel):
            raise GenerateError(f"`{rel}` looks like a secret or key file and will not be read.")
        if not validate_file_path(rel, base_dir=root):
            raise GenerateError(f"`{rel}` is outside the repository.")
        path = root / rel
        if not path.is_file():
            raise GenerateError(f"`{rel}` was not found on this branch.")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            raise GenerateError(f"`{rel}` is not a readable text file.")
        if len(text) > MAX_FILE_CHARS:
            raise GenerateError(
                f"`{rel}` is too large ({len(text):,} characters; limit {MAX_FILE_CHARS:,})."
            )
        total += len(text)
        if total > MAX_TOTAL_CODE_CHARS:
            raise GenerateError(
                f"The code files are too large together (limit {MAX_TOTAL_CODE_CHARS:,} characters). "
                "Name fewer files."
            )
        files.append((rel, text))
    return files


def load_template(doc_type, repo_root, cfg):
    """Return (template_path, text) from the configured page templates, or ("", "")."""
    templates = (cfg or {}).get("page-templates") or {}
    rel = templates.get(doc_type)
    if not rel:
        return "", ""
    root = Path(repo_root)
    if not validate_file_path(rel, base_dir=root) or not (root / rel).is_file():
        raise GenerateError(f"Template for `{doc_type}` is configured as `{rel}` but was not found.")
    return rel, (root / rel).read_text(encoding="utf-8")[:MAX_TEMPLATE_CHARS]


# =============================================================================
# GENERATION
# =============================================================================


def build_prompt(page, code_files, template_path, template, style, notes):
    parts = [
        f"Write the first version of the documentation page `{page}` from the code below.",
        "Rules:\n"
        "- Describe only behaviour the code shows. Do not invent endpoints, fields, limits, "
        "permissions or error codes.\n"
        "- Treat docstrings and comments as claims, not facts. When a statement comes only "
        "from a docstring or comment and the code does not enforce it, still write it if it "
        "matters to the reader, but list it under `verify`.\n"
        "- When the template asks for something the code does not show (for example the "
        "owner or required permissions), keep the template's placeholder and list it under "
        "`verify`.\n"
        "- Follow the template's front matter, headings and section order exactly. Remove "
        "template comments meant only for authors.\n"
        "- Leave `last_reviewed` as it is; it is set by code.",
        'Reply with JSON only: {"page": "<complete file content>", "verify": ["<item to check>", ...]}',
    ]
    if template:
        parts.append(f"Template (`{template_path}`):\n{template}")
    if style:
        parts.append(f"Documentation style rules:\n{style}")
    if notes:
        parts.append(f"Reviewer notes (highest priority):\n{notes}")
    for rel, text in code_files:
        parts.append(f"Code file `{rel}`:\n```\n{text}\n```")
    return "\n\n".join(parts)


def parse_model_reply(text):
    """Return (page_content, verify_list). Raises GenerateError."""
    text = strip_code_fences((text or "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise GenerateError("The model did not return the expected JSON.")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        raise GenerateError("The model returned invalid JSON.")
    page = data.get("page")
    if not isinstance(page, str) or not page.strip():
        raise GenerateError("The model returned an empty page.")
    page = strip_code_fences(page.strip()) + "\n"
    verify = [str(v).strip() for v in data.get("verify") or [] if str(v).strip()]
    return page, verify


def generate_page(page, code_files, template_path, template, style, notes):
    """Call the model. Model and network errors propagate so the job fails."""
    response = get_client().chat.completions.create(
        model=get_model_name(),
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_prompt(page, code_files, template_path, template, style, notes),
            },
        ],
    )
    content, verify = parse_model_reply(response.choices[0].message.content)
    ok, errors = validate_format(content, page)
    if not ok:
        raise GenerateError(f"The generated page failed format checks: {errors}")
    return content, verify


# =============================================================================
# OUTPUT
# =============================================================================


def build_comment(written, verify, template_path, code_paths, error=None):
    lines = ["## 📝 Generated documentation draft", ""]
    if error:
        lines += [f"❌ **Could not generate the page.** {error}", "", USAGE]
        return "\n".join(lines)
    lines.append(
        "This is a **first draft written from code**. Review every statement before merging."
    )
    lines.append("")
    lines.append(f"**Code read:** {', '.join(f'`{c}`' for c in code_paths)}")
    if template_path:
        lines.append(f"**Template:** `{template_path}`")
    lines.append("")
    lines.append("**Files written:**")
    lines += [f"- ✅ `{p}`" for p, _ in written]
    if verify:
        lines += ["", "### ⚠️ Check before merging", ""]
        lines += [f"- [ ] {v}" for v in verify]
    for path, content in written:
        lines += ["", f"<details><summary><b>{path}</b></summary>", "", "```markdown"]
        lines += [content.rstrip("\n"), "```", "", "</details>"]
    return "\n".join(lines)


def post_comment(pr_number, body):
    gh_token = os.environ.get("GH_TOKEN")
    if not gh_token or not pr_number or pr_number == "unknown":
        print(body)
        return
    msg_file = Path("/tmp/generate_docs_comment.md")
    msg_file.write_text(body, encoding="utf-8")
    run_command_safe(
        ["gh", "pr", "comment", str(pr_number), "--body-file", str(msg_file)],
        env={**os.environ, "GH_TOKEN": gh_token},
        check=False,
    )


def commit_and_push(docs_subfolder, paths, pr_branch):
    """Commit the written pages and push them to the PR branch."""
    os.chdir("..")
    files = [f"{docs_subfolder}/{p}" for p in paths]
    run_command_safe(["git", "add"] + files, check=True)
    run_command_safe(
        ["git", "commit", "-m", "docs: generate first draft from code\n\nAssisted-by: code-to-docs AI"],
        check=True,
    )
    run_command_safe(["git", "push", "origin", f"HEAD:refs/heads/{pr_branch}"], check=True)


# =============================================================================
# ENTRY POINT
# =============================================================================


def run_generate(comment_body, style_guidelines, setup_docs_environment, resolve_push_target,
                 dry_run=False):
    """Run [generate-docs]. Exits non-zero on any failure."""
    pr_number = os.environ.get("PR_NUMBER", "unknown")
    repo_root = os.getcwd()
    request = None
    try:
        cfg = load_repo_config() or {}
        pairs = language_pairs(cfg)
        request = parse_request(comment_body)
        validate_page_path(request["page"], pairs)
        doc_type = request["template"] or doc_type_of(request["page"], pairs)
        code_files = read_code_files(request["code"], repo_root)
        template_path, template = load_template(doc_type, repo_root, cfg)

        docs_subfolder = os.environ.get("DOCS_SUBFOLDER")
        if not docs_subfolder:
            raise GenerateError("[generate-docs] needs `docs-subfolder` (same-repo docs).")
        if not setup_docs_environment():
            raise GenerateError(f"Docs folder `{docs_subfolder}` was not found.")
        if os.path.exists(request["page"]) and not request["overwrite"]:
            raise GenerateError(
                f"`{request['page']}` already exists. Add `overwrite: yes` to replace it."
            )

        print(f"Generating {request['page']} from {', '.join(request['code'])}")
        content, verify = generate_page(
            request["page"], code_files, template_path, template, style_guidelines,
            request["notes"],
        )
        results = [(request["page"], "", content)]

        counterpart, _ = counterpart_of(request["page"], pairs)
        if counterpart and in_parity_scope(request["page"], cfg.get("parity-scope")):
            print(f"Generating counterpart {counterpart}")
            translated = generate_new_page(
                "", counterpart, style_guidelines, reference=(request["page"], content)
            )
            if not translated:
                raise GenerateError(f"Could not generate the counterpart `{counterpart}`.")
            results.append((counterpart, "", translated))

        results = stamp_review_dates(results, cfg=cfg)
        written = [(path, updated) for path, _, updated in results]

        if dry_run:
            print(build_comment(written, verify, template_path, request["code"]))
            return

        for path, updated in written:
            if not write_page(path, updated):
                raise GenerateError(f"Could not write `{path}`.")
        pr_branch, _, merged = resolve_push_target(pr_number)
        if merged or not pr_branch:
            raise GenerateError("The pull request is merged or its branch could not be found.")
        commit_and_push(docs_subfolder, [p for p, _ in written], pr_branch)
        post_comment(pr_number, build_comment(written, verify, template_path, request["code"]))
        print("✅ Generated draft committed to the pull request")
    except GenerateError as e:
        print(f"❌ {e}")
        post_comment(pr_number, build_comment([], [], "", [], error=str(e)))
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        msg = f"Git command failed: {sanitize_output(str(e))}"
        print(f"❌ {msg}")
        post_comment(pr_number, build_comment([], [], "", [], error=msg))
        sys.exit(1)
    except Exception as e:
        msg = f"Model or runtime error: {sanitize_output(str(e))}"
        print(f"❌ {msg}")
        post_comment(pr_number, build_comment([], [], "", [], error=msg))
        sys.exit(1)
