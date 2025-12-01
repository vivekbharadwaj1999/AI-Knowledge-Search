import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
)
from app.ingest import ingest_file, UPLOAD_DIR
from app.qa import answer_question
from app.vector_store import list_documents, clear_vector_store
from app.config import GROQ_MODEL
from app.insights import generate_insights
from app.report import generate_document_report
from app.relations import analyze_cross_document_relations
from app.critique import run_critique

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
    )
    return CritiqueResponse(**data)


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
