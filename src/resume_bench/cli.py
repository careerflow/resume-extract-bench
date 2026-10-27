from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from resume_bench.settings import settings

app = typer.Typer(
    name="resume-bench",
    help="ResumeBench - benchmark for structured resume extraction",
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
):
    """Grade extraction results against ground truth (ResumeBench-exact scoring)."""
    from resume_bench.grading.grader import grade_pipelines_exact

    console.print(f"Grading {len(pipelines)} pipeline(s)...")

    reports = grade_pipelines_exact(
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


@app.command()
def leaderboard(
    split: str = typer.Option("test", help="Dataset split"),
):
    """Show the ResumeBench-exact leaderboard."""
    _show_exact_leaderboard(split)


def _show_exact_leaderboard(split: str) -> None:
    """Print a ResumeBench-exact leaderboard from saved grade files."""
    from resume_bench.grading.models import CellCounts

    output_dir = settings.output_dir
    rows: list[tuple[str, CellCounts, int]] = []

    for pipeline_dir in sorted(output_dir.iterdir()):
        grades_dir = pipeline_dir / split / "grades_exact"
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
        console.print("[yellow]No ResumeBench grades found. Run 'grade' first.[/yellow]")
        return

    rows.sort(key=lambda r: r[1].f1, reverse=True)

    table = Table(title=f"ResumeBench-Exact Leaderboard ({split})")
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
):
    """Grade a predictions JSONL file against ground truth (ResumeBench-exact).

    Each line should be: {"resume_id": "...", "prediction": {...}}
    """
    from resume_bench.dataset.loader import load_split
    from resume_bench.grading.grader import grade_single_exact
    from resume_bench.grading.models import CellCounts

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
                pred = {k: v for k, v in record.items() if k != "resume_id"}
            predictions[rid] = pred

    rb_scores = []

    for rid, gt in gt_by_id.items():
        pred = predictions.get(rid)
        if pred is None:
            console.print(f"  [yellow]Missing prediction for {rid}[/yellow]")
            continue
        score = grade_single_exact(gt, pred)
        score.resume_id = rid
        rb_scores.append(score)

    if not rb_scores:
        console.print("[red]No predictions matched any ground truth resume IDs.[/red]")
        return

    agg = CellCounts(
        correct=sum(s.cells.correct for s in rb_scores),
        expected=sum(s.cells.expected for s in rb_scores),
        predicted=sum(s.cells.predicted for s in rb_scores),
    )
    resume_f1s = [s.cells.f1 for s in rb_scores]
    macro_f1 = sum(resume_f1s) / len(resume_f1s)

    table = Table(title=f"ResumeBench-Exact Results ({len(rb_scores)} resumes)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Micro F1", f"{agg.f1:.4f}")
    table.add_row("Macro F1", f"{macro_f1:.4f}")
    table.add_row("Micro Precision", f"{agg.precision:.4f}")
    table.add_row("Micro Recall", f"{agg.recall:.4f}")
    table.add_row("Cells Correct", str(agg.correct))
    table.add_row("Cells Expected", str(agg.expected))
    table.add_row("Cells Predicted", str(agg.predicted))
    table.add_row("Resumes Graded", f"{len(rb_scores)} / {len(gt_by_id)}")

    console.print(table)


@app.command()
def report(
    split: str = typer.Option("test", help="Dataset split"),
    pipelines: Optional[list[str]] = typer.Option(None, help="Filter to specific pipelines"),
):
    """Generate leaderboard report from graded results."""
    console.print("[yellow]Report generation uses 'leaderboard' command.[/yellow]")
    _show_exact_leaderboard(split)


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
