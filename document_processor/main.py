import io
import logging
import os
from pathlib import Path

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .chunker import DocumentChunker
from .cleaner import clean_text


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
app = FastAPI(title="Document Processor", version="0.2.0")
MAX_DOCUMENT_BYTES = int(os.getenv("MAX_DOCUMENT_BYTES", str(512 * 1024 * 1024)))


def extract_text_from_pdf(content: bytes) -> tuple[str, int, str]:
    page_texts: list[str] = []
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            for page in document:
                text = clean_text(page.get_text("text"))
                if text:
                    page_texts.append(text)
            pages = document.page_count
        if page_texts:
            return "\n\n".join(page_texts), pages, "pymupdf"
    except Exception:
        logger.exception("PyMuPDF extraction failed; trying pdfplumber")

    import pdfplumber

    with pdfplumber.open(io.BytesIO(content)) as document:
        page_texts = [clean_text(page.extract_text() or "") for page in document.pages]
        page_texts = [text for text in page_texts if text]
        return "\n\n".join(page_texts), len(document.pages), "pdfplumber"


def decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="Unsupported text encoding")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "document_processor"}


@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000, ge=200, le=4000),
    overlap: int = Form(150, ge=0, le=1000),
    strategy: str = Form("recursive", pattern="^(recursive|sentence|paragraph)$"),
) -> dict:
    if overlap >= chunk_size:
        raise HTTPException(status_code=422, detail="overlap must be smaller than chunk_size")
    content = await file.read(MAX_DOCUMENT_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(content) > MAX_DOCUMENT_BYTES:
        max_document_mib = MAX_DOCUMENT_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Файл превышает допустимый размер {max_document_mib} МБ",
        )

    filename = Path(file.filename or "unknown").name
    extension = Path(filename).suffix.lower()
    try:
        if extension == ".pdf":
            text, pages, method = extract_text_from_pdf(content)
        elif extension in {".txt", ".md"}:
            text, pages, method = clean_text(decode_text(content)), 1, "plain_text"
        else:
            raise HTTPException(status_code=415, detail="Supported formats: PDF, TXT and MD")

        if len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="No readable text was extracted")
        chunks = DocumentChunker(chunk_size=chunk_size, overlap=overlap).chunk_text(text, strategy)
        return {
            "filename": filename,
            "pages": pages,
            "chunks": len(chunks),
            "total_chars": len(text),
            "method": method,
            "chunks_data": chunks,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document processing failed")
        raise HTTPException(status_code=422, detail="Could not parse the document") from exc


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "Document Processor", "version": "0.2.0"}
