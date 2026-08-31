from __future__ import annotations

from pathlib import Path

import fitz


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
