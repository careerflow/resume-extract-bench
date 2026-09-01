from __future__ import annotations

import csv
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from resume_bench.schema.sections import SECTIONS
from resume_bench.settings import settings

console = Console()


def _collect_graded_summaries(split: str, pipeline_names: list[str] | None = None) -> dict[str, dict]:
    """Collect graded summary data from output directories."""
    base = settings.output_dir

    if not base.exists():
        return {}

    candidates = []

    if pipeline_names:
        candidates = pipeline_names
    else:
        for p in base.iterdir():
            if p.is_dir() and (p / split / "grades").exists():
                candidates.append(p.name)

    summaries = {}

    for name in candidates:
        pipeline_dir = base / name / split
        grades_dir = pipeline_dir / "grades"

        if not grades_dir.exists():
            continue

        grade_files = list(grades_dir.glob("*.grade.json"))

        if not grade_files:
            continue

        macro_f1s = []
        section_f1s: dict[str, list[float]] = {}
        section_omission: dict[str, list[float]] = {}
        section_hallucination: dict[str, list[float]] = {}

        for gf in grade_files:
            with open(gf) as f:
                data = json.load(f)

            macro_f1s.append(data.get("macro_entity_f1", 0.0))

            for sec_name, sec_data in data.get("sections", {}).items():
                if not sec_data.get("is_vacuous", False):
                    section_f1s.setdefault(sec_name, []).append(sec_data.get("f1", 0.0))
                    section_omission.setdefault(sec_name, []).append(sec_data.get("omission_rate", 0.0))
                    section_hallucination.setdefault(sec_name, []).append(sec_data.get("hallucination_rate", 0.0))

        avg_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0

        avg_sections = {}
        avg_omission = {}
        avg_hallucination = {}

        for sec_name, vals in section_f1s.items():
            avg_sections[sec_name] = round(sum(vals) / len(vals), 4)

        for sec_name, vals in section_omission.items():
            avg_omission[sec_name] = round(sum(vals) / len(vals), 4)

        for sec_name, vals in section_hallucination.items():
            avg_hallucination[sec_name] = round(sum(vals) / len(vals), 4)

        result_files = list(pipeline_dir.glob("*.result.json"))
        total_latency_ms = 0
        total_cost = 0.0
        completed = 0
        errors = 0
        cost_count = 0

        for rf in result_files:
            with open(rf) as f:
                rdata = json.load(f)

            if rdata.get("error"):
                errors += 1
            else:
                completed += 1

            total_latency_ms += rdata.get("latency_ms", 0)

            cost = rdata.get("cost_usd")
            if cost is not None:
                total_cost += cost
                cost_count += 1

        total_resumes = completed + errors
        avg_latency_s = (total_latency_ms / total_resumes / 1000) if total_resumes else 0.0
        avg_cost = (total_cost / cost_count) if cost_count else None

        summaries[name] = {
            "resume_entity_f1": round(avg_f1, 4),
            "section_f1": avg_sections,
            "section_omission": avg_omission,
            "section_hallucination": avg_hallucination,
            "graded": len(grade_files),
            "completed": completed,
            "errors": errors,
            "completion_rate": round(completed / total_resumes, 4) if total_resumes else 0.0,
            "avg_latency_s": round(avg_latency_s, 2),
            "avg_cost_usd": round(avg_cost, 5) if avg_cost is not None else None,
            "total_cost_usd": round(total_cost, 4) if cost_count else None,
        }

    return summaries


def print_leaderboard(split: str = "test") -> None:
    """Print leaderboard table to console."""
    summaries = _collect_graded_summaries(split)

    if not summaries:
        console.print("[yellow]No graded results found. Run 'resume-bench grade' first.[/yellow]")
        return

    ranked = sorted(summaries.items(), key=lambda x: x[1]["resume_entity_f1"], reverse=True)

    table = Table(title=f"ResumeExtractBench Leaderboard ({split})")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Pipeline")
    table.add_column("Entity F1", justify="right")
    table.add_column("Basics", justify="right")
    table.add_column("Done", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("$/Resume", justify="right")
    table.add_column("Resumes", justify="right")

    for rank, (name, data) in enumerate(ranked, 1):
        completion = f"{data['completion_rate']:.0%}"
        latency = f"{data['avg_latency_s']:.1f}s"
        cost = f"${data['avg_cost_usd']:.4f}" if data['avg_cost_usd'] is not None else "-"

        basics_val = data["section_f1"].get("basics")
        basics = f"{basics_val:.3f}" if basics_val is not None else "-"

        row = [
            str(rank),
            name,
            f"{data['resume_entity_f1']:.4f}",
            basics,
            completion,
            latency,
            cost,
            str(data["graded"]),
        ]

        table.add_row(*row)

    console.print(table)


def generate_reports(
    split: str = "test",
    pipeline_names: list[str] | None = None,
    html: bool = False,
) -> Path:
    """Generate CSV (and optionally HTML) leaderboard reports."""
    summaries = _collect_graded_summaries(split, pipeline_names)

    output_path = settings.output_dir / "reports" / split
    output_path.mkdir(parents=True, exist_ok=True)

    ranked = sorted(summaries.items(), key=lambda x: x[1]["resume_entity_f1"], reverse=True)

    csv_path = output_path / "leaderboard.csv"
    fieldnames = ["rank", "pipeline", "entity_f1", "completion_rate", "avg_latency_s", "avg_cost_usd", "graded"]
    fieldnames += [spec.name for spec in SECTIONS]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, (name, data) in enumerate(ranked, 1):
            row = {
                "rank": rank,
                "pipeline": name,
                "entity_f1": f"{data['resume_entity_f1']:.4f}",
                "completion_rate": f"{data['completion_rate']:.4f}",
                "avg_latency_s": f"{data['avg_latency_s']:.2f}",
                "avg_cost_usd": f"{data['avg_cost_usd']:.5f}" if data['avg_cost_usd'] is not None else "",
                "graded": data["graded"],
            }

            for spec in SECTIONS:
                val = data["section_f1"].get(spec.name)
                row[spec.name] = f"{val:.4f}" if val is not None else ""

            writer.writerow(row)

    summary_path = output_path / "summary.json"

    with open(summary_path, "w") as f:
        json.dump(
            {"split": split, "pipelines": dict(ranked)},
            f, indent=2,
        )

    if html:
        html_path = output_path / "leaderboard.html"
        _write_html_report(ranked, split, html_path)

    return output_path


def _write_html_report(
    ranked: list[tuple[str, dict]],
    split: str,
    html_path: Path,
) -> None:
    """Write a simple HTML leaderboard."""
    section_names = [spec.name for spec in SECTIONS]

    header_cells = "".join(f"<th>{s}</th>" for s in section_names)

    rows_html = ""

    for rank, (name, data) in enumerate(ranked, 1):
        section_cells = ""

        for s in section_names:
            val = data["section_f1"].get(s)
            section_cells += f"<td>{val:.3f}</td>" if val is not None else "<td>-</td>"

        cost = f"${data['avg_cost_usd']:.4f}" if data['avg_cost_usd'] is not None else "-"

        rows_html += f"""<tr>
            <td>{rank}</td>
            <td>{name}</td>
            <td><strong>{data['resume_entity_f1']:.4f}</strong></td>
            <td>{data['completion_rate']:.0%}</td>
            <td>{data['avg_latency_s']:.1f}s</td>
            <td>{cost}</td>
            {section_cells}
        </tr>\n"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ResumeExtractBench Leaderboard - {split}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 1400px; margin: 2rem auto; padding: 0 1rem; }}
        h1 {{ color: #1a1a2e; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
        th {{ background: #1a1a2e; color: white; }}
        tr:nth-child(even) {{ background: #f8f8f8; }}
        tr:first-child td {{ font-weight: bold; background: #e8f5e9; }}
        td:nth-child(2) {{ text-align: left; }}
    </style>
</head>
<body>
    <h1>ResumeExtractBench Leaderboard</h1>
    <p>Split: <strong>{split}</strong></p>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Pipeline</th>
                <th>Entity F1</th>
                <th>Completion</th>
                <th>Latency</th>
                <th>Cost/Resume</th>
                {header_cells}
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>"""

    html_path.write_text(html_content)
