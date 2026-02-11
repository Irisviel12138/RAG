"""Document parsers for PDF, DOCX, and PPTX files."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List



@dataclass
class ParsedDocument:
    doc_id: str
    content: str
    source_type: str


def _read_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    pages: List[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[Page {i}]\n{text}")
    return "\n\n".join(pages)


def _read_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(BytesIO(data))
    parts: List[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _read_pptx(data: bytes) -> str:
    from pptx import Presentation

    prs = Presentation(BytesIO(data))
    slides: List[str] = []
    for idx, slide in enumerate(prs.slides, start=1):
        chunk: List[str] = []
        for shape in slide.shapes:
            text = getattr(shape, "text", "")
            text = text.strip() if text else ""
            if text:
                chunk.append(text)
        if chunk:
            slides.append(f"[Slide {idx}]\n" + "\n".join(chunk))
    return "\n\n".join(slides)


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    ext = Path(filename).suffix.lower()
    doc_id = Path(filename).stem

    if ext == ".pdf":
        content = _read_pdf(data)
        return ParsedDocument(doc_id=doc_id, content=content, source_type="pdf")

    if ext == ".docx":
        content = _read_docx(data)
        return ParsedDocument(doc_id=doc_id, content=content, source_type="docx")

    if ext == ".pptx":
        content = _read_pptx(data)
        return ParsedDocument(doc_id=doc_id, content=content, source_type="pptx")

    raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .docx, or .pptx")
