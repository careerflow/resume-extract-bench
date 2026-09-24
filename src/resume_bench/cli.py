from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from resume_bench.settings import settings

app = typer.Typer(
    name="resume-bench",
    help="ResumeExtractBench - benchmark for structured resume extraction",
    no_args_is_help=True,
)
console = Console()


@app.command()
def download(
    split: str = typer.Option("all", help="Split to download: dev, test, all"),
    revision: str = typer.Option("main", help="Dataset revision tag"),
):
    """Download the benchmark dataset from HuggingFace."""
    from resume_bench.dataset.loader import download_dataset

    console.print(f"Downloading split={split}, revision={revision}...")
    path = download_dataset(split=split, revision=revision)
    console.print(f"Dataset downloaded to {path}")


@app.command()
def status():
    """Show dataset status and counts."""
    from resume_bench.dataset.loader import get_dataset_status

    info = get_dataset_status()

    table = Table(title="Dataset Status")
    table.add_column("Split")
    table.add_column("Count", justify="right")
    table.add_column("Path")

    for split_name, split_info in info.items():
        table.add_row(split_name, str(split_info["count"]), str(split_info["path"]))

    console.print(table)


@app.command()
def providers(
    check: bool = typer.Option(False, "--check", help="Run healthcheck on each provider"),
):
    """List available pipelines and providers."""
    from resume_bench.providers.pipelines import PIPELINES

    table = Table(title="Available Pipelines")
    table.add_column("Pipeline")
    table.add_column("Provider")
    table.add_column("Input Mode")
    table.add_column("Notes")

    for spec in PIPELINES:
        table.add_row(
            spec.pipeline_name,
            spec.provider_name,
            spec.input_mode.value,
            spec.notes or "",
        )

    console.print(table)


@app.command()
def run(
    pipelines: list[str] = typer.Argument(..., help="Pipeline names to run"),
    split: str = typer.Option("test", help="Dataset split to run on"),
    limit: Optional[int] = typer.Option(None, help="Max resumes to process"),
    concurrency: int = typer.Option(4, help="Max concurrent extractions"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached results"),
):
    """Run extraction pipelines on the dataset."""
    from resume_bench.runner.runner import run_pipelines

    console.print(f"Running {len(pipelines)} pipeline(s) on {split} split...")

    results = run_pipelines(
        pipeline_names=pipelines,
        split=split,
        limit=limit,
        concurrency=concurrency,
        use_cache=not no_cache,
    )

    for name, stats in results.items():
        console.print(
            f"  {name}: {stats['extracted']} extracted, "
            f"{stats['cached']} cached, {stats['errors']} errors"
        )


@app.command()
def grade(
    pipelines: list[str] = typer.Argument(..., help="Pipeline names to grade"),
    split: str = typer.Option("test", help="Dataset split"),
    threshold: float = typer.Option(0.5, help="Alignment similarity threshold"),
    extractbench: bool = typer.Option(
        False, "--extractbench",
        help="Use ExtractBench-exact scoring (binary exact match, cell-level P/R/F1)",
    ),
):
    """Grade extraction results against ground truth."""
    if extractbench:
        from resume_bench.grading.grader import grade_pipelines_extractbench

        console.print(f"Grading {len(pipelines)} pipeline(s) with ExtractBench-exact scoring...")

        reports = grade_pipelines_extractbench(
            pipeline_names=pipelines,
            split=split,
        )

        for name, report in reports.items():
            console.print(f"\n[bold]{name}[/bold]")
            console.print(f"  Micro F1:  {report['micro_f1']:.4f}")
            console.print(f"  Macro F1:  {report['macro_f1']:.4f}")
            console.print(
                f"  Cells:     {report['total_cells']['correct']}"
                f" / {report['total_cells']['expected']} expected"
                f" / {report['total_cells']['predicted']} predicted"
            )
            console.print(f"  Completed: {report['completed']} / {report['total_resumes']}")
    else:
        from resume_bench.grading.grader import grade_pipelines

        console.print(f"Grading {len(pipelines)} pipeline(s)...")

        reports = grade_pipelines(
            pipeline_names=pipelines,
            split=split,
            threshold=threshold,
        )

        for name, report in reports.items():
            console.print(f"\n[bold]{name}[/bold]")
            console.print(f"  Resume Entity F1: {report['resume_entity_f1']:.4f}")
            console.print(f"  Completion rate:  {report['completion_rate']:.1%}")


@app.command()
def report(
    split: str = typer.Option("test", help="Dataset split"),
    pipelines: Optional[list[str]] = typer.Option(None, help="Filter to specific pipelines"),
    html: bool = typer.Option(False, "--html", help="Generate HTML report"),
):
    """Generate leaderboard and reports from graded results."""
    from resume_bench.report.leaderboard import generate_reports

    console.print("Generating reports...")

    output_path = generate_reports(split=split, pipeline_names=pipelines, html=html)
    console.print(f"Reports written to {output_path}")


@app.command()
def leaderboard(
    split: str = typer.Option("test", help="Dataset split"),
    extractbench: bool = typer.Option(
        False, "--extractbench",
        help="Show ExtractBench-exact leaderboard from grades_extractbench/",
    ),
):
    """Show the current leaderboard."""
    if extractbench:
        _show_extractbench_leaderboard(split)
    else:
        from resume_bench.report.leaderboard import print_leaderboard

        print_leaderboard(split=split)


def _show_extractbench_leaderboard(split: str) -> None:
    """Print a simple ExtractBench-exact leaderboard from saved grade files."""
    import json

    from resume_bench.grading.models import CellCounts

    output_dir = settings.output_dir
    rows: list[tuple[str, CellCounts, int]] = []

    for pipeline_dir in sorted(output_dir.iterdir()):
        grades_dir = pipeline_dir / split / "grades_extractbench"
        if not grades_dir.exists():
            continue

        agg = CellCounts()
        count = 0
        for grade_path in grades_dir.glob("*.grade.json"):
            with open(grade_path) as f:
                data = json.load(f)
            if not data.get("completed", True):
                continue
            agg.correct += data["correct"]
            agg.expected += data["expected"]
            agg.predicted += data["predicted"]
            count += 1

        if count:
            rows.append((pipeline_dir.name, agg, count))

    if not rows:
        console.print("[yellow]No ExtractBench grades found. Run 'grade --extractbench' first.[/yellow]")
        return

    rows.sort(key=lambda r: r[1].f1, reverse=True)

    table = Table(title=f"ExtractBench-Exact Leaderboard ({split})")
    table.add_column("Rank", justify="right")
    table.add_column("Pipeline")
    table.add_column("Micro F1", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Resumes", justify="right")

    for i, (name, agg, count) in enumerate(rows, 1):
        table.add_row(
            str(i),
            name,
            f"{agg.f1:.4f}",
            f"{agg.precision:.4f}",
            f"{agg.recall:.4f}",
            str(count),
        )

    console.print(table)


@app.command()
def grade_file(
    predictions_path: Path = typer.Argument(..., help="JSONL file with predictions"),
    split: str = typer.Option("test", help="Dataset split for ground truth"),
    threshold: float = typer.Option(0.5, help="Alignment similarity threshold"),
    extractbench: bool = typer.Option(
        False, "--extractbench",
        help="Use ExtractBench-exact scoring (binary exact match, cell-level P/R/F1)",
    ),
):
    """Grade a predictions JSONL file against ground truth.

    Each line should be: {"resume_id": "...", "prediction": {...}}
    """
    import json

    from resume_bench.dataset.loader import load_split

    cases = load_split(split)
    gt_by_id = {c.resume_id: c.ground_truth for c in cases}

    predictions = {}

    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            rid = record["resume_id"]
            pred = record.get("prediction") or record.get("output")
            if pred is None:
                # Flat format: prediction fields are top-level in the record
                pred = {k: v for k, v in record.items() if k != "resume_id"}
            predictions[rid] = pred

    if extractbench:
        from resume_bench.grading.grader import grade_single_extractbench
        from resume_bench.grading.models import CellCounts

        eb_scores = []

        for rid, gt in gt_by_id.items():
            pred = predictions.get(rid)
            if pred is None:
                console.print(f"  [yellow]Missing prediction for {rid}[/yellow]")
                continue
            score = grade_single_extractbench(gt, pred)
            score.resume_id = rid
            eb_scores.append(score)

        if not eb_scores:
            console.print("[red]No predictions matched any ground truth resume IDs.[/red]")
            return

        agg = CellCounts(
            correct=sum(s.cells.correct for s in eb_scores),
            expected=sum(s.cells.expected for s in eb_scores),
            predicted=sum(s.cells.predicted for s in eb_scores),
        )
        resume_f1s = [s.cells.f1 for s in eb_scores]
        macro_f1 = sum(resume_f1s) / len(resume_f1s)

        table = Table(title=f"ExtractBench-Exact Results ({len(eb_scores)} resumes)")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Micro F1", f"{agg.f1:.4f}")
        table.add_row("Macro F1", f"{macro_f1:.4f}")
        table.add_row("Micro Precision", f"{agg.precision:.4f}")
        table.add_row("Micro Recall", f"{agg.recall:.4f}")
        table.add_row("Cells Correct", str(agg.correct))
        table.add_row("Cells Expected", str(agg.expected))
        table.add_row("Cells Predicted", str(agg.predicted))
        table.add_row("Resumes Graded", f"{len(eb_scores)} / {len(gt_by_id)}")

        console.print(table)

    else:
        from resume_bench.grading.grader import grade_single
        from resume_bench.grading.models import GradingConfig

        cfg = GradingConfig(threshold=threshold)
        scores = []

        for rid, gt in gt_by_id.items():
            pred = predictions.get(rid)

            if pred is None:
                console.print(f"  [yellow]Missing prediction for {rid}[/yellow]")
                continue

            score = grade_single(gt, pred, cfg)
            score.resume_id = rid
            scores.append(score)

        if not scores:
            console.print("[red]No predictions matched any ground truth resume IDs.[/red]")
            return

        avg_f1 = sum(s.macro_entity_f1 for s in scores) / len(scores)
        avg_basics = sum(s.basics_field_accuracy for s in scores) / len(scores)

        table = Table(title=f"Grade Results ({len(scores)} resumes)")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Headline Entity F1", f"{avg_f1:.4f}")
        table.add_row("Basics Accuracy", f"{avg_basics:.4f}")
        table.add_row("Resumes Graded", f"{len(scores)} / {len(gt_by_id)}")

        section_f1s: dict[str, list[float]] = {}

        for s in scores:
            for name, sec in s.sections.items():
                if not sec.is_vacuous:
                    section_f1s.setdefault(name, []).append(sec.f1)

        for name, vals in sorted(section_f1s.items(), key=lambda x: -sum(x[1]) / len(x[1])):
            avg = sum(vals) / len(vals)
            table.add_row(f"  {name}", f"{avg:.4f}")

        console.print(table)


@app.command()
def validate(
    dataset_path: Path = typer.Argument(..., help="Path to dataset JSONL file"),
):
    """Validate a dataset file against the resume schema."""
    from resume_bench.dataset.validate import validate_dataset

    errors = validate_dataset(dataset_path)

    if errors:
        console.print(f"[red]Validation failed with {len(errors)} errors:[/red]")
        for err in errors[:10]:
            console.print(f"  {err}")
    else:
        console.print("[green]Validation passed![/green]")


if __name__ == "__main__":
    app()
