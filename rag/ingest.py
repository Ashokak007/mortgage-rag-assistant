"""Load PDFs and split them into overlapping text chunks with source metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str  # file name
    page: int    # 1-based page number

    def to_dict(self) -> dict:
        return asdict(self)


def clean_text(text: str) -> str:
    """Normalise whitespace and re-join words hyphenated across line breaks."""
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into ~chunk_size-character windows that end on a sentence or
    word boundary where possible, with `overlap` characters shared between
    neighbouring chunks so answers that straddle a boundary are not lost."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = text.strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if cut < chunk_size * 0.5:          # no good sentence break: use a word break
                cut = window.rfind(" ")
            if cut > 0:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        # don't start mid-word
        while start < len(text) and start > 0 and text[start - 1] != " ":
            start += 1
    return chunks


def load_pdf_chunks(pdf_path: Path, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    reader = PdfReader(str(pdf_path))
    out: list[Chunk] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        for piece in split_into_chunks(text, chunk_size, overlap):
            out.append(Chunk(text=piece, source=pdf_path.name, page=page_no))
    return out


def load_folder(folder: Path, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {folder}. Add some documents first.")
    chunks: list[Chunk] = []
    for pdf in pdfs:
        chunks.extend(load_pdf_chunks(pdf, chunk_size, overlap))
    return chunks
