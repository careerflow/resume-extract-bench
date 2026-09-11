from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

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
):
    """Grade extraction results against ground truth."""
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
):
    """Show the current leaderboard."""
    from resume_bench.report.leaderboard import print_leaderboard

    print_leaderboard(split=split)


@app.command()
def grade_file(
    predictions_path: Path = typer.Argument(..., help="JSONL file with predictions"),
    split: str = typer.Option("test", help="Dataset split for ground truth"),
    threshold: float = typer.Option(0.5, help="Alignment similarity threshold"),
):
    """Grade a predictions JSONL file against ground truth.

    Each line should be: {"resume_id": "...", "prediction": {...}}
    """
    import json

    from resume_bench.dataset.loader import load_split
    from resume_bench.grading.grader import grade_single
    from resume_bench.grading.models import GradingConfig

    cases = load_split(split)
    gt_by_id = {c.resume_id: c.ground_truth for c in cases}
    cfg = GradingConfig(threshold=threshold)

    predictions = {}

    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            rid = record["resume_id"]
            pred = record.get("prediction", record.get("output", {}))
            predictions[rid] = pred

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
