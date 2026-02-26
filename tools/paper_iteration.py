#!/usr/bin/env python3
"""Iterate sigma-TAP workspace against paper-derived requirements.

This script builds a concrete iteration report by checking whether current
workspace documents cover key requirements extracted from the TAP papers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

ROOT = Path(__file__).resolve().parents[1]

PAPERS = [
    "TAPequation-FINAL.pdf",
    "Applications-of-TAP.pdf",
    "Paper1-FINAL.pdf",
    "Paper2-FINAL.pdf",
    "Long-run patterns in the discovery of the adjacent possible.pdf",
]

DOCS = [
    "README.md",
    "reviews/adjacent-possible-alignment-review.md",
    "TASKS.md",
    "ENGINEERING_PLAN.md",
]

@dataclass
class Requirement:
    id: str
    description: str
    keywords: List[str]

REQUIREMENTS = [
    Requirement("R1", "Treat TAP as a variant family (baseline/two-scale/logistic/differential)", ["variant", "two-scale", "logistic", "differential"]),
    Requirement("R2", "Model/report regime transitions (plateau/acceleration/explosive/extinction)", ["plateau", "regime", "explosive", "extinction", "blow-up"]),
    Requirement("R3", "Integrate long-run empirical diagnostics (rate scaling/distribution/diversification)", ["long-run", "scaling", "distribution", "diversification", "zipf", "heaps"]),
    Requirement("R4", "Include mixed explanatory framing for Type III systems (mechanistic + functional)", ["type iii", "functional", "mechanistic", "non-ergodic"]),
    Requirement("R5", "Enforce claim traceability (paper refs + artifact refs)", ["claim", "citation", "artifact", "supported", "exploratory"]),
]


def read_pdf_text(path: Path) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(str(path))
        chunks = []
        for page in reader.pages[:8]:  # sample first 8 pages for speed
            chunks.append((page.extract_text() or "").lower())
        return "\n".join(chunks)
    except Exception:
        return ""


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return ""


def keyword_hits(text: str, keywords: List[str]) -> Dict[str, int]:
    return {k: text.count(k.lower()) for k in keywords}


def main() -> None:
    paper_hits = {}
    for name in PAPERS:
        text = read_pdf_text(ROOT / name)
        paper_hits[name] = {req.id: keyword_hits(text, req.keywords) for req in REQUIREMENTS}

    docs_text = {name: read_text(ROOT / name) for name in DOCS}

    req_eval = []
    for req in REQUIREMENTS:
        doc_scores = {}
        total = 0
        for name, text in docs_text.items():
            hits = sum(keyword_hits(text, req.keywords).values())
            doc_scores[name] = hits
            total += hits
        status = "covered" if total >= 10 else "partial" if total >= 4 else "missing"
        req_eval.append({
            "id": req.id,
            "description": req.description,
            "status": status,
            "total_keyword_hits": total,
            "doc_scores": doc_scores,
        })

    priority_actions = []
    for r in req_eval:
        if r["status"] != "covered":
            priority_actions.append({
                "requirement": r["id"],
                "action": f"Implement concrete artifact/tests for {r['description']}",
                "status": r["status"],
            })

    report = {
        "papers_checked": PAPERS,
        "docs_checked": DOCS,
        "requirement_evaluation": req_eval,
        "priority_actions": priority_actions,
    }

    out_json = ROOT / "outputs/paper_iteration_report.json"
    out_md = ROOT / "outputs/paper_iteration_report.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Paper Iteration Report (Run 001)",
        "",
        "## Requirement status",
        "",
    ]
    for r in req_eval:
        lines.append(f"- **{r['id']}** {r['description']}: `{r['status']}` (hits={r['total_keyword_hits']})")
    lines += ["", "## Priority actions", ""]
    if priority_actions:
        for a in priority_actions:
            lines.append(f"- {a['requirement']}: {a['action']} (`{a['status']}`)")
    else:
        lines.append("- None. All requirements covered by current docs.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
