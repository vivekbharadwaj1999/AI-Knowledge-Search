import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import json
from pathlib import Path

from app.schemas import (
    AskRequest,
    AskResponse,
    InsightsRequest,
    InsightsResponse,
    ReportRequest,
    DocumentReport,
    CrossDocRelationsRequest,
    CrossDocRelations,
    CritiqueRequest,
    CritiqueResponse,
    CritiqueLogResponse,
    CritiqueLogRow,
)
from app.ingest import ingest_file, UPLOAD_DIR
from app.qa import answer_question
from app.vector_store import list_documents, clear_vector_store
from app.config import GROQ_MODEL
from app.insights import generate_insights
from app.report import generate_document_report
from app.relations import analyze_cross_document_relations
from app.critique import (
    run_critique,
    reset_critique_log_file,
)
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Knowledge Search Engine")

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "ok", "message": "AI Knowledge Search backend running"}


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    allowed_exts = (".pdf", ".txt", ".csv", ".docx", ".pptx", ".xlsx")
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: PDF, TXT, CSV, DOCX, PPTX, XLSX.",
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    chunk_count = ingest_file(file_path, doc_name=file.filename)
    if chunk_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in file",
        )

    return {"status": "ok", "chunks_indexed": chunk_count}


@app.post("/ask", response_model=AskResponse)
async def ask_question_route(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer, chunks, sources = answer_question(
        payload.question,
        k=payload.top_k,
        doc_name=payload.doc_name,
        model=payload.model,
    )

    model_used = payload.model or GROQ_MODEL

    return AskResponse(
        answer=answer,
        context=chunks,
        model_used=model_used,
        sources=sources,
    )


@app.post("/insights", response_model=InsightsResponse)
async def insights_route(payload: InsightsRequest):
    if not payload.answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty")

    data = generate_insights(
        question=payload.question,
        answer=payload.answer,
        context=payload.context,
        model=payload.model,
    )
    return InsightsResponse(**data)


@app.get("/documents")
async def get_documents():
    return {"documents": list_documents()}


@app.post("/report", response_model=DocumentReport)
def create_report(req: ReportRequest):
    try:
        report = generate_document_report(
            doc_name=req.doc_name,
            model=req.model,
        )
        return report
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print("Error generating report:", e)
        raise HTTPException(
            status_code=500, detail="Failed to generate document report"
        )


@app.post("/document-relations", response_model=CrossDocRelations)
async def document_relations_route(payload: CrossDocRelationsRequest):
    try:
        result = analyze_cross_document_relations(model=payload.model)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("Error in /document-relations:", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze document relations",
        )


@app.post("/critique", response_model=CritiqueResponse)
async def critique_route(payload: CritiqueRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    data = run_critique(
        question=payload.question,
        answer_model=payload.answer_model,
        critic_model=payload.critic_model,
        top_k=payload.top_k,
        doc_name=payload.doc_name,
        self_correct=payload.self_correct,
        similarity=payload.similarity,
    )
    return data


LOG_PATH = Path("data/critique_log.jsonl")


@app.get("/critique-log-rows", response_model=CritiqueLogResponse)
def get_critique_log_rows():
    """
    Return one row per critique run for frontend export.
    Reads data/critique_log.jsonl written by run_critique().
    """
    if not LOG_PATH.exists():
        return CritiqueLogResponse(rows=[])

    rows: list[CritiqueLogRow] = []

    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            rounds = obj.get("rounds") or []
            if not rounds:
                continue

            r1 = rounds[0]
            rN = rounds[-1]

            s1 = (r1.get("scores") or {}) or {}
            sN = (rN.get("scores") or {}) or {}

            def num(d, key):
                v = d.get(key)
                try:
                    return float(v) if v is not None else None
                except Exception:
                    return None

            r1_corr = num(s1, "correctness")
            rN_corr = num(sN, "correctness")
            r1_hal = num(s1, "hallucination_risk")
            rN_hal = num(sN, "hallucination_risk")

            row = CritiqueLogRow(
                timestamp=obj.get("timestamp"),
                question=obj.get("question"),
                answer_model=obj.get("answer_model"),
                critic_model=obj.get("critic_model"),
                doc_name=obj.get("doc_name"),
                self_correct=bool(obj.get("self_correct")),
                similarity=obj.get("similarity"),
                num_rounds=len(rounds),
                r1_correctness=r1_corr,
                rN_correctness=rN_corr,
                r1_hallucination=r1_hal,
                rN_hallucination=rN_hal,
                delta_correctness=(
                    rN_corr - r1_corr
                    if r1_corr is not None and rN_corr is not None
                    else None
                ),
                delta_hallucination=(
                    rN_hal - r1_hal
                    if r1_hal is not None and rN_hal is not None
                    else None
                ),
            )
            rows.append(row)

    return CritiqueLogResponse(rows=rows)


@app.post("/reset-critique-log")
def reset_critique_log_endpoint():
    """
    Reset / delete all logged critique runs (JSONL file).
    """
    reset_critique_log_file()
    return {"status": "ok", "message": "Critique log reset"}


@app.delete("/documents")
def delete_all_documents():
    if os.path.exists(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except Exception as e:
                    print("Delete failed:", e)

    clear_vector_store()

    return {"status": "ok", "message": "All documents removed"}
