import io

from fastapi import UploadFile, HTTPException
from pypdf import PdfReader


async def extract_pdf(file: UploadFile):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    pdf_bytes = await file.read()

    try:
        reader = PdfReader(
            io.BytesIO(pdf_bytes)
        )

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read PDF file."
        )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        pages.append({
            "page": page_number,
            "text": text,
        })

    return pages