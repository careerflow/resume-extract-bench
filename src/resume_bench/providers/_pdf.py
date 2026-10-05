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

    doc = fitz.open(str(pdf_path))
    image_paths = []

    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        img_path = Path(tempfile.mktemp(suffix=f"_page{i}.png"))
        pix.save(str(img_path))
        image_paths.append(img_path)

    doc.close()

    return image_paths
