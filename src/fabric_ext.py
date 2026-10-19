"""
Fabric extensions to code-to-docs.

Three additions, all switched on from `.code-to-docs/config.json` on the base branch:

1. Language parity ("language-pairs": [["en", "ar"]])
   When a page in one language is selected, its counterpart in the other language
   (same relative path) is selected too. The counterpart is generated after the
   source page and receives the updated source page as reference, so both carry
   the same facts.

2. New pages ("allow-new-pages": true, "max-new-pages": 1)
   The model may propose a new page when a change adds a user-facing capability
   that no existing page is the right home for. New pages appear in the review
   list like any other file and are created only after the reviewer keeps them
   checked and comments [update-docs].

3. Review dates ("review-date-field": "last_reviewed", "review-date-timezone": "UTC")
   The front-matter review date of every changed page is set by code, not by the
   model, to today's date in the configured time zone.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from config import check_context_error, get_client, get_model_name, load_repo_config
from generation import (
    _SYSTEM_PROMPT,
    ask_ai_for_updated_content,
    generate_updates_parallel,
    load_full_content,
    overwrite_file,
    strip_code_fences,
    validate_format,
)
from security_utils import sanitize_output, validate_docs_file_extension, validate_file_path

DOC_EXTENSIONS = (".md", ".adoc", ".rst")
_SAFE_NEW_PATH = re.compile(r"^[a-z0-9][a-z0-9_\-/]*\.(md|adoc|rst)$")
MAX_PAGE_PREVIEW_CHARS = 600
MAX_TEMPLATE_CHARS = 6000


# =============================================================================
# CONFIGURATION
# =============================================================================


def _config():
    cfg = load_repo_config()
    return cfg if isinstance(cfg, dict) else {}


def language_pairs(cfg=None):
    """Return [(primary, secondary), ...] from config, ignoring malformed entries."""
    cfg = _config() if cfg is None else cfg
    pairs = []
    for item in cfg.get("language-pairs") or []:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and all(isinstance(x, str) and x and "/" not in x and ".." not in x for x in item)
        ):
            pairs.append((item[0], item[1]))
    return pairs


def counterpart_of(path, pairs):
    """Return (counterpart_path, is_secondary) for a page, or (None, False)."""
    parts = Path(path).parts
    if len(parts) < 2:
        return None, False
    for primary, secondary in pairs:
        if parts[0] == primary:
            return str(Path(secondary, *parts[1:])), False
        if parts[0] == secondary:
            return str(Path(primary, *parts[1:])), True
    return None, False


# =============================================================================
# SELECTION
# =============================================================================


def list_doc_pages(root="."):
    """List documentation pages under root, skipping hidden folders such as .doc-index."""
    pages = []
    root_path = Path(root)
    for path in sorted(root_path.rglob("*")):
        rel = path.relative_to(root_path)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.is_file() and path.suffix in DOC_EXTENSIONS:
            pages.append(str(rel))
    return pages


def _first_heading(path):
    text = load_full_content(path)
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def is_valid_new_page_path(path, existing_pages, pairs):
    """A proposed new page must be a safe, new, lower-case doc path in a known folder."""
    if not isinstance(path, str) or not _SAFE_NEW_PATH.match(path):
        return False
    if ".." in Path(path).parts or path in existing_pages or Path(path).exists():
        return False
    if not validate_file_path(path) or not validate_docs_file_extension(path):
        return False
    if pairs and Path(path).parts[0] not in {p for p, _ in pairs}:
        return False
    parent = str(Path(path).parent)
    return any(str(Path(p).parent) == parent for p in existing_pages)


def propose_new_pages(diff, selected, style_guidelines="", cfg=None):
    """Ask the model whether the change needs a new page. Returns a list of paths."""
    cfg = _config() if cfg is None else cfg
    if not cfg.get("allow-new-pages"):
        return []
    max_new = cfg.get("max-new-pages", 1)
    if not isinstance(max_new, int) or max_new < 1:
        return []

    pairs = language_pairs(cfg)
    existing = list_doc_pages()
    primary_langs = {p for p, _ in pairs}
    listing = "\n".join(
        f"- {p}: {_first_heading(p)}"
        for p in existing
        if not primary_langs or Path(p).parts[0] in primary_langs
    )

    prompt = f"""A code change is shown below, followed by the list of existing documentation
pages and the pages already selected for an update.

Decide whether the change adds a user-facing capability (for example a new API endpoint,
a new command or a new workflow) whose usage steps have no suitable home in any existing
page. Most changes need no new page: if an existing page can hold the content, answer with
an empty list.

If a new page is needed, propose at most {max_new} page(s):
- Put it in the existing folder for its page type (for example how-to/ for task steps).
- Use the naming style of the existing pages (lower-case, hyphens, same extension).
{"- Use only these language folders: " + ", ".join(sorted(primary_langs)) if primary_langs else ""}

Answer with JSON only, no prose:
{{"new_pages": [{{"path": "<relative path>", "reason": "<one sentence>"}}]}}

Code change:
{diff[:60000]}

Existing pages (path: title):
{listing}

Already selected for update:
{chr(10).join("- " + s for s in selected) or "- none"}

Documentation style rules:
{style_guidelines[:8000]}
"""
    try:
        response = get_client().chat.completions.create(
            model=get_model_name(),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        raw = strip_code_fences((response.choices[0].message.content or "").strip())
    except Exception as e:
        check_context_error(e)
        print(f"Warning: new-page proposal failed: {sanitize_output(str(e))}")
        return []

    try:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start : end + 1]) if start >= 0 else {}
    except json.JSONDecodeError:
        print("Warning: new-page proposal was not valid JSON, ignoring")
        return []

    proposals = []
    for item in data.get("new_pages") or []:
        path = (item or {}).get("path", "").strip().removeprefix("./")
        if is_valid_new_page_path(path, existing, pairs) and path not in proposals:
            print(f"New page proposed: {path} ({item.get('reason', '')})")
            proposals.append(path)
        else:
            print(f"Ignoring invalid new-page proposal: {path!r}")
    return proposals[:max_new]


def add_language_counterparts(files, pairs):
    """Add each page's counterpart when it exists, or when the source page is new."""
    result = list(dict.fromkeys(files))
    for path in list(result):
        counterpart, _ = counterpart_of(path, pairs)
        if not counterpart or counterpart in result:
            continue
        if Path(counterpart).exists() or not Path(path).exists():
            print(f"Language parity: adding {counterpart} (counterpart of {path})")
            result.append(counterpart)
    return result


def extend_selection(diff, files, style_guidelines=""):
    """Apply new-page proposals and language parity to the discovered files."""
    cfg = _config()
    files = list(dict.fromkeys(files or []))
    files += [p for p in propose_new_pages(diff, files, style_guidelines, cfg) if p not in files]
    pairs = language_pairs(cfg)
    if pairs and cfg.get("enforce-language-parity", True):
        files = add_language_counterparts(files, pairs)
    return files


# =============================================================================
# GENERATION
# =============================================================================


def _template_for(path):
    """Return an existing sibling page to copy structure and style from."""
    folder = Path(path).parent
    for candidate in sorted(folder.glob(f"*{Path(path).suffix}")):
        if candidate.is_file():
            return str(candidate), load_full_content(str(candidate))[:MAX_TEMPLATE_CHARS]
    return "", ""


def generate_new_page(
    diff, path, style_guidelines="", instructions="", pr_description="", reference=None
):
    """Write a complete new page. Returns its content, or None."""
    template_path, template = _template_for(path)
    parts = [
        f"Write the complete content of a NEW documentation page `{path}`.",
        "Output only the raw file content, with no code fences around it.",
    ]
    if reference:
        src_path, src_content = reference
        parts.append(
            f"This page is the counterpart of `{src_path}` in another language. Write the same "
            "content in this page's language, following the style rules for that language. "
            f"Keep code, field names and URLs unchanged.\n\nSource page:\n{src_content}"
        )
    else:
        parts.append(
            "Document only what the code change below adds, for the reader of this page type. "
            "Do not invent behaviour that the change does not show."
        )
        parts.append(f"Code change:\n{diff[:60000]}")
    if pr_description:
        parts.append(f"PR description (context only):\n{pr_description}")
    if template:
        parts.append(
            f"Existing page from the same folder (`{template_path}`). Match its front matter "
            f"fields, heading levels, section order and example style:\n{template}"
        )
    if style_guidelines:
        parts.append(f"Documentation style rules (follow them):\n{style_guidelines}")
    if instructions:
        parts.append(f"Reviewer instructions (highest priority):\n{instructions}")
    prompt = "\n\n".join(parts)

    try:
        response = get_client().chat.completions.create(
            model=get_model_name(),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        output = strip_code_fences((response.choices[0].message.content or "").strip())
    except Exception as e:
        check_context_error(e)
        print(f"Warning: could not generate new page {path}: {sanitize_output(str(e))}")
        return None

    if not output or output.strip() == "NO_UPDATE_NEEDED":
        return None
    if not output.endswith("\n"):
        output += "\n"
    is_valid, errors = validate_format(output, path)
    if not is_valid:
        print(f"Warning: new page {path} failed format validation: {errors}")
        return None
    return output


def _parity_instruction(src_path, updated_src):
    return (
        f"This page is the counterpart of `{src_path}` in another language. That page was just "
        "updated as shown below. Apply the same factual changes here, in this page's language, "
        "and leave the rest of this page's wording as it is. Updated counterpart:\n"
        f"{updated_src}"
    )


def generate_all(
    diff,
    files,
    max_workers=5,
    user_instructions="",
    file_instructions=None,
    style_guidelines="",
    pr_description="",
):
    """
    Generate updates for existing pages and content for new pages.

    Counterpart pages in the secondary language are generated after their source page
    and receive the updated source as reference.

    Returns:
        list of (file_path, original_content, updated_content); original is "" for new pages.
    """
    from comments import _resolve_file_instructions

    file_instructions = dict(file_instructions or {})
    pairs = language_pairs()

    def instructions_for(path):
        parts = [user_instructions, _resolve_file_instructions(path, file_instructions)]
        return "\n".join(p for p in parts if p)

    second_pass = []
    first_pass = []
    for path in files:
        counterpart, is_secondary = counterpart_of(path, pairs)
        if is_secondary and counterpart in files:
            second_pass.append((path, counterpart))
        else:
            first_pass.append(path)

    results = {}

    existing_first = [p for p in first_pass if Path(p).exists()]
    new_first = [p for p in first_pass if not Path(p).exists()]
    for path, original, updated in generate_updates_parallel(
        diff,
        existing_first,
        max_workers=max_workers,
        user_instructions=user_instructions,
        file_instructions=file_instructions,
        style_guidelines=style_guidelines,
        pr_description=pr_description,
    ):
        results[path] = (path, original, updated)
    for path in new_first:
        content = generate_new_page(
            diff, path, style_guidelines, instructions_for(path), pr_description
        )
        if content:
            results[path] = (path, "", content)

    for path, source in second_pass:
        source_result = results.get(source)
        if Path(path).exists():
            current = load_full_content(path)
            if not current:
                continue
            per_file = dict(file_instructions)
            if source_result:
                extra = _parity_instruction(source, source_result[2])
                existing_note = _resolve_file_instructions(path, file_instructions)
                per_file[path] = f"{existing_note}\n{extra}".strip()
            updated = ask_ai_for_updated_content(
                diff,
                path,
                current,
                user_instructions=user_instructions,
                file_instructions=per_file,
                style_guidelines=style_guidelines,
                pr_description=pr_description,
            )
            if updated.strip() != "NO_UPDATE_NEEDED":
                results[path] = (path, current, updated)
        elif source_result:
            content = generate_new_page(
                diff,
                path,
                style_guidelines,
                instructions_for(path),
                pr_description,
                reference=(source, source_result[2]),
            )
            if content:
                results[path] = (path, "", content)

    return [results[p] for p in files if p in results]


# =============================================================================
# REVIEW DATES AND WRITING
# =============================================================================


def today_in(tz_name):
    """Today's date as YYYY-MM-DD in the given IANA time zone (UTC if unknown)."""
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    return datetime.now(tz).strftime("%Y-%m-%d")


def set_front_matter_field(content, field, value):
    """Set `field: value` inside a leading YAML front-matter block, if the field exists."""
    if not content.startswith("---"):
        return content
    end = content.find("\n---", 3)
    if end == -1:
        return content
    head, rest = content[:end], content[end:]
    pattern = re.compile(rf"^({re.escape(field)}\s*:\s*).*$", re.MULTILINE)
    if not pattern.search(head):
        return content
    return pattern.sub(lambda m: f"{m.group(1)}{value}", head, count=1) + rest


def stamp_review_dates(files_with_content, cfg=None):
    """Set the review-date field of every changed page to today, by code."""
    cfg = _config() if cfg is None else cfg
    field = cfg.get("review-date-field")
    if not field or not isinstance(field, str):
        return files_with_content
    today = today_in(cfg.get("review-date-timezone") or "UTC")
    stamped = []
    for path, original, updated in files_with_content:
        stamped.append((path, original, set_front_matter_field(updated, field, today)))
    return stamped


def write_page(path, content):
    """Write a page, creating its folder for new pages, with the usual safety checks."""
    if not validate_file_path(path) or not validate_docs_file_extension(path):
        print(f"❌ Security: Invalid file path rejected: {path}")
        return False
    try:
        os.makedirs(Path(path).parent, exist_ok=True)
    except OSError as e:
        print(f"Failed to create folder for {path}: {sanitize_output(str(e))}")
        return False
    return overwrite_file(path, content)


def fabric_enabled(cfg=None):
    """True when any Fabric option is set in the repo config."""
    try:
        cfg = _config() if cfg is None else cfg
    except Exception:
        return False
    keys = ("language-pairs", "allow-new-pages", "review-date-field")
    return any(cfg.get(k) for k in keys)
