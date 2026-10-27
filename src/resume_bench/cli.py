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
    verbatim: bool = typer.Option(True, "--verbatim/--no-verbatim", help="Verbatim mode (case/punctuation sensitive); on by default"),
):
    """Grade extraction results against ground truth (ResumeBench-Strict scoring)."""
    from resume_bench.grading.grader import grade_pipelines_strict

    mode = "verbatim" if verbatim else "standard"
    console.print(f"Grading {len(pipelines)} pipeline(s) [mode={mode}]...")

    reports = grade_pipelines_strict(
        pipeline_names=pipelines,
        split=split,
        verbatim=verbatim,
    )

    for name, report in reports.items():
        console.print(f"\n[bold]{name}[/bold]")
        console.print(f"  Macro F1:  {report['macro_f1']:.4f}")
        console.print(
            f"  Units:     {report['total_units']['correct']}"
            f" / {report['total_units']['expected']} expected"
            f" / {report['total_units']['predicted']} predicted"
        )
        console.print(f"  Completed: {report['completed']} / {report['total_resumes']}")


@app.command()
def leaderboard(
    split: str = typer.Option("test", help="Dataset split"),
):
    """Show the ResumeBench-Strict leaderboard."""
    output_dir = settings.output_dir
    rows: list[tuple[str, float, int]] = []

    for pipeline_dir in sorted(output_dir.iterdir()):
        grades_dir = pipeline_dir / split / "grades_strict"
        if not grades_dir.exists():
            continue

        f1s: list[float] = []
        for grade_path in grades_dir.glob("*.grade.json"):
            with open(grade_path) as f:
                data = json.load(f)
            if not data.get("completed", True):
                continue
            f1s.append(data.get("f1", 0.0))

        if f1s:
            macro_f1 = sum(f1s) / len(f1s)
            rows.append((pipeline_dir.name, macro_f1, len(f1s)))

    if not rows:
        console.print("[yellow]No ResumeBench-Strict grades found. Run 'grade' first.[/yellow]")
        return

    rows.sort(key=lambda r: r[1], reverse=True)

    table = Table(title=f"ResumeBench-Strict Leaderboard ({split})")
    table.add_column("Rank", justify="right")
    table.add_column("Pipeline")
    table.add_column("Macro F1", justify="right")
    table.add_column("Resumes", justify="right")

    for i, (name, f1, count) in enumerate(rows, 1):
        table.add_row(str(i), name, f"{f1:.4f}", str(count))

    console.print(table)


@app.command()
def grade_file(
    predictions_path: Path = typer.Argument(..., help="JSONL file with predictions"),
    split: str = typer.Option("test", help="Dataset split for ground truth"),
    verbatim: bool = typer.Option(True, "--verbatim/--no-verbatim", help="Verbatim mode (case/punctuation sensitive); on by default"),
):
    """Grade a predictions JSONL file against ground truth (ResumeBench-Strict).

    Each line should be: {"resume_id": "...", "prediction": {...}}
    """
    from resume_bench.dataset.loader import load_split
    from resume_bench.grading.grader import grade_single_strict

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

    results: list[dict] = []

    for rid, gt in gt_by_id.items():
        pred = predictions.get(rid)
        if pred is None:
            console.print(f"  [yellow]Missing prediction for {rid}[/yellow]")
            continue
        result = grade_single_strict(gt, pred, verbatim=verbatim)
        result["resume_id"] = rid
        results.append(result)

    if not results:
        console.print("[red]No predictions matched any ground truth resume IDs.[/red]")
        return

    resume_f1s = [r["f1"] for r in results]
    macro_f1 = sum(resume_f1s) / len(resume_f1s)

    total_correct = sum(r["correct"] for r in results)
    total_expected = sum(r["expected"] for r in results)
    total_predicted = sum(r["predicted"] for r in results)

    mode = "verbatim" if verbatim else "standard"
    table = Table(title=f"ResumeBench-Strict Results ({len(results)} resumes, {mode})")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Macro F1", f"{macro_f1:.4f}")
    table.add_row("Units Correct", str(total_correct))
    table.add_row("Units Expected", str(total_expected))
    table.add_row("Units Predicted", str(total_predicted))
    table.add_row("Resumes Graded", f"{len(results)} / {len(gt_by_id)}")

    console.print(table)


@app.command()
def report(
    split: str = typer.Option("test", help="Dataset split"),
    pipelines: Optional[list[str]] = typer.Option(None, help="Filter to specific pipelines"),
):
    """Generate leaderboard report from graded results."""
    console.print("[yellow]Report generation uses 'leaderboard' command.[/yellow]")
    leaderboard(split=split)


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
