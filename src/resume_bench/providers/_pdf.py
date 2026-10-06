from __future__ import annotations

from pathlib import Path

import fitz

from resume_bench.providers.base import DocumentInfo


def get_document_info(pdf_path: Path) -> DocumentInfo:
    """Capture document metrics from a PDF file."""
    file_size = pdf_path.stat().st_size

    doc = fitz.open(str(pdf_path))
    page_count = len(doc)

    char_count = 0
    word_count = 0
    for page in doc:
        text = page.get_text()
        char_count += len(text)
        word_count += len(text.split())
    doc.close()

    return DocumentInfo(
        page_count=page_count,
        file_size_bytes=file_size,
        char_count=char_count,
        word_count=word_count,
        token_count_estimate=char_count // 4 if char_count else 0,
    )


def pdf_to_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))

    parts = []
    for page in doc:
        parts.append(page.get_text())

    doc.close()

    return "\n".join(parts)


def pdf_to_images(pdf_path: Path, dpi: int = 150) -> list[Path]:
    """Render each PDF page as a PNG image. Returns list of temp file paths."""
    import tempfile

    from pdf2image import convert_from_path

    pil_images = convert_from_path(str(pdf_path), dpi=dpi)
    image_paths = []

    for i, img in enumerate(pil_images):
        img_path = Path(tempfile.mktemp(suffix=f"_page{i}.png"))
        img.save(str(img_path), "PNG")
        image_paths.append(img_path)

    return image_paths


def pdf_to_base64_images(pdf_path: Path, dpi: int = 150) -> list[str]:
    """Rasterize each PDF page to a base64-encoded PNG string in memory.

    Uses ``pdf2image`` (poppler-based) for rasterization, matching the
    approach used by ExtractBench.  Unlike ``pdf_to_images()``, this does
    not write temp files — the PNG bytes are returned directly as base64
    strings suitable for embedding in API requests as
    ``data:image/png;base64,...`` URIs.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Rendering resolution (default 150).

    Returns:
        List of base64-encoded PNG strings, one per page.
    """
    import base64
    import io

    from pdf2image import convert_from_path

    pil_images = convert_from_path(str(pdf_path), dpi=dpi)
    images: list[str] = []
    for img in pil_images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return images


def pdf_to_markdown_llamaparse(
    pdf_path: Path,
    tier: str = "agentic",
    api_key: str | None = None,
) -> tuple[str, dict]:
    """Parse a PDF to markdown using LlamaParse.

    Args:
        pdf_path: Path to the PDF file.
        tier: LlamaParse tier (default ``"cost_effective"``).
        api_key: LlamaParse API key.  If not provided, reads from
            ``RESUME_BENCH_LLAMA_CLOUD_API_KEY`` environment variable.

    Returns:
        Tuple of (markdown string, parse metadata dict).

    Raises:
        ValueError: If no API key is available.
        RuntimeError: If the parsing job fails.
    """
    import os
    import time

    from llama_cloud import LlamaCloud

    if api_key is None:
        api_key = os.getenv("RESUME_BENCH_LLAMA_CLOUD_API_KEY", "")
    if not api_key:
        raise ValueError(
            "LlamaParse API key not set. Set RESUME_BENCH_LLAMA_CLOUD_API_KEY "
            "or pass api_key parameter."
        )

    client = LlamaCloud(token=api_key)
    start = time.perf_counter()

    with open(pdf_path, "rb") as f:
        upload = client.files.upload_file(upload_file=f, project_id=None)

    job = client.parsing.create_parsing_job(
        input_file_id=upload.id,
        config={"result_type": "markdown"},
    )

    job_id = job.id
    while True:
        status = client.parsing.get_parsing_job(job_id)
        if status.status == "SUCCESS":
            break
        elif status.status in ("ERROR", "FAILED", "CANCELLED"):
            raise RuntimeError(f"LlamaParse job {job_id} failed: {status.status}")
        time.sleep(2)

    result = client.parsing.get_parsing_job_result(job_id)
    markdown = result.markdown if hasattr(result, "markdown") else str(result)

    latency = round(time.perf_counter() - start, 2)
    parse_meta = {
        "job_id": job_id,
        "tier": tier,
        "parse_latency_seconds": latency,
    }

    return markdown, parse_meta
