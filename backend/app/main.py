import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AskRequest, AskResponse, InsightsRequest, InsightsResponse
from app.ingest import ingest_file, UPLOAD_DIR
from app.qa import answer_question
from app.vector_store import list_documents, clear_vector_store
from app.config import GROQ_MODEL
from app.insights import generate_insights

app = FastAPI(title="AI Knowledge Search Engine")

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # MUST NOT include "*"
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

    answer, chunks = answer_question(
        payload.question,
        k=payload.top_k,
        doc_name=payload.doc_name,
        model=payload.model,
    )

    model_used = payload.model or GROQ_MODEL

    return AskResponse(answer=answer, context=chunks, model_used=model_used)


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
    # Pydantic will validate / coerce the dict into InsightsResponse
    return InsightsResponse(**data)


@app.get("/documents")
async def get_documents():
    return {"documents": list_documents()}


@app.delete("/documents")
def delete_all_documents():
    # delete uploaded files
    if os.path.exists(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except Exception as e:
                    print("Delete failed:", e)

    # delete JSONL vector store
    clear_vector_store()

    return {"status": "ok", "message": "All documents removed"}


