"""/ingest — upload one or more files and push them through the RAG ingestion."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile

from ..rag import ingest_paths

api = APIRouter()


@api.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="amplify-upload-"))
    saved: List[Path] = []
    try:
        for f in files:
            dst = tmp / (f.filename or "uploaded.bin")
            with dst.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(dst)
        return ingest_paths(saved)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
