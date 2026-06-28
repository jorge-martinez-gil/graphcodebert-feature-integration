"""Automatic report generation: Markdown + publication-quality LaTeX tables.

Tables and reports are generated from results, never hand-written, so that every
number in the documentation is traceable to a run and regenerated on demand.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

DEFAULT_COLUMNS = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc", "ece"]


def _fmt(v, nd: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def markdown_table(rows: Sequence[Dict], columns: Sequence[str], index_col: str = "method") -> str:
    cols = [index_col] + list(columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [head, sep]
    for r in rows:
        cells = [str(r.get(index_col, ""))] + [_fmt(r.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_table(
    rows: Sequence[Dict],
    columns: Sequence[str],
    index_col: str = "method",
    caption: str = "Benchmark results.",
    label: str = "tab:results",
    bold_best: bool = True,
) -> str:
    """Render a ``booktabs`` LaTeX table; bolds the best value per column."""
    cols = list(columns)
    best = {}
    if bold_best:
        for c in cols:
            vals = [(i, r.get(c)) for i, r in enumerate(rows) if isinstance(r.get(c), (int, float))]
            if vals:
                # lower is better for calibration/error columns
                better = min if c in {"ece", "brier"} else max
                best[c] = better(vals, key=lambda t: t[1])[0]
    align = "l" + "c" * len(cols)
    out = [
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{align}}}",
        "    \\toprule",
        "    " + " & ".join([index_col.title()] + [c.replace("_", "-").upper() for c in cols]) + " \\\\",
        "    \\midrule",
    ]
    for i, r in enumerate(rows):
        cells = [str(r.get(index_col, ""))]
        for c in cols:
            v = r.get(c, "")
            s = _fmt(v) if isinstance(v, (int, float)) else str(v)
            if bold_best and best.get(c) == i:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        out.append("    " + " & ".join(cells) + " \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def write_tables(rows: Sequence[Dict], out_dir: str, columns: Sequence[str] = DEFAULT_COLUMNS,
                 caption: str = "Benchmark results.", label: str = "tab:results") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    cols = [c for c in columns if any(c in r for r in rows)]
    md = markdown_table(rows, cols)
    tex = latex_table(rows, cols, caption=caption, label=label)
    md_path = os.path.join(out_dir, "results_table.md")
    tex_path = os.path.join(out_dir, "results_table.tex")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    return {"markdown": md_path, "latex": tex_path}


def write_report(run_dir: str, results: Dict, manifest: Dict, figures: Optional[List[str]] = None) -> str:
    """Write a human-readable Markdown benchmark report for one run."""
    os.makedirs(run_dir, exist_ok=True)
    cfg = manifest.get("config", {})
    rows = results.get("rows", [])
    cols = [c for c in DEFAULT_COLUMNS if any(c in r for r in rows)]
    lines = [
        f"# Benchmark report — {cfg.get('name', 'run')}",
        "",
        f"- **Dataset:** `{cfg.get('data', {}).get('name', '?')}`",
        f"- **Backend:** `{cfg.get('model', {}).get('backend', '?')}`",
        f"- **Features:** {', '.join(cfg.get('features', [])) or '(none)'}",
        f"- **Seed:** {manifest.get('seed')}  |  **Config hash:** `{manifest.get('config_hash')}`",
        f"- **Git commit:** `{manifest.get('git_commit')}`",
        f"- **Generated:** {manifest.get('timestamp_utc')}",
        "",
        "## Results",
        "",
        markdown_table(rows, cols) if rows else "_no rows_",
        "",
    ]
    if results.get("significance"):
        lines += ["## Statistical significance", "", "```json",
                  json.dumps(results["significance"], indent=2), "```", ""]
    if figures:
        lines += ["## Figures", ""]
        for fig in figures:
            rel = os.path.relpath(fig, run_dir)
            lines.append(f"![{os.path.basename(fig)}]({rel})")
        lines.append("")
    lines += ["---", "_Generated automatically by FeatFuse. Every number is reproducible "
              "via the accompanying `manifest.json`._"]
    path = os.path.join(run_dir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
