"""Check source syntax, local documentation links and retained duplicate copies."""

import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    sources = sorted(ROOT.glob("*.py"))
    for directory in ("src", "scripts", "tests"):
        sources.extend(sorted((ROOT / directory).rglob("*.py")))
    for path in sources:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(str(exc))

    documents = sorted(ROOT.glob("*.md"))
    for directory in ("docs", "data", "examples", "results", "requirements"):
        documents.extend(sorted((ROOT / directory).rglob("*.md")))
    documents.append(ROOT / "artifacts/README.md")
    link_count = 0
    for path in documents:
        text = path.read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in link or link.startswith(("#", "mailto:")):
                continue
            target = unquote(link.split("#", 1)[0])
            link_count += 1
            if not (path.parent / target).exists():
                errors.append(f"Broken local link in {path.relative_to(ROOT)}: {link}")
        if text.count("```") % 2:
            errors.append(f"Unclosed code fence: {path.relative_to(ROOT)}")

    manifest = json.loads((ROOT / "docs/removed-duplicates.json").read_text())
    for entry in manifest:
        kept = ROOT / entry["retained_copy"]
        # Text blobs are stored with LF in Git; Windows checkout may use CRLF.
        content = kept.read_bytes().replace(b"\r\n", b"\n") if kept.is_file() else None
        if content is None or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            errors.append(f"Missing or changed retained duplicate: {entry['retained_copy']}")

    for path in [ROOT / "requirements.txt", *(ROOT / "requirements").glob("*.txt")]:
        for line in path.read_text().splitlines():
            if line.startswith("-r ") and not (path.parent / line[3:].strip()).is_file():
                errors.append(f"Missing requirements include in {path.name}: {line}")

    for error in errors:
        print(error, file=sys.stderr)
    print(f"Checked {len(sources)} Python files, {link_count} local links, {len(manifest)} retained copies; {len(errors)} errors.")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
