#!/usr/bin/env python3
import json
import os
import sys

MAX_ITEMS = int(os.getenv("MAX_SARIF_ITEMS", "50"))

def esc(s: str) -> str:
    # Basic Markdown table escaping
    return (s or "").replace("\n", " ").replace("|", "\\|")

def main():
    if len(sys.argv) != 2:
        print("Usage: sarif_to_md.py path/to/report.sarif", file=sys.stderr)
        return 2

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        sarif = json.load(f)

    runs = sarif.get("runs", [])
    rows = []
    total = 0

    for run in runs:
        results = run.get("results", []) or []
        for r in results:
            total += 1
            rule_id = r.get("ruleId") or r.get("rule", {}).get("id") or ""
            level = r.get("level") or r.get("kind") or ""
            msg = (r.get("message") or {}).get("text") or ""
            loc = ""
            locs = r.get("locations") or []
            if locs:
                pl = (locs[0].get("physicalLocation") or {})
                uri = ((pl.get("artifactLocation") or {}).get("uri")) or ""
                region = pl.get("region") or {}
                line = region.get("startLine")
                loc = f"{uri}:{line}" if uri and line else (uri or "")
            rows.append((rule_id, level, msg, loc))

    header = "### SARIF findings\n"
    if total == 0:
        md = header + "\nNo findings in SARIF report.\n"
        print(md)
        return 0

    md = [header, "", f"Total findings: **{total}**", ""]
    md.append("| Rule | Level | Message | Location |")
    md.append("|---|---|---|---|")

    shown = 0
    for rule_id, level, msg, loc in rows[:MAX_ITEMS]:
        md.append(f"| `{esc(rule_id)}` | {esc(level)} | {esc(msg)} | `{esc(loc)}` |")
        shown += 1

    if total > shown:
        md.append("")
        md.append(f"_Showing first {shown} findings (set `MAX_SARIF_ITEMS` to adjust)._")

    # Add a marker so we can “update” the same comment later
    md.append("")
    md.append("<!-- sarif-comment-marker -->")

    print("\n".join(md))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
