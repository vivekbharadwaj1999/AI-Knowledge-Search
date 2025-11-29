import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AskRequest, AskResponse
from app.ingest import ingest_file, UPLOAD_DIR   # ✅ use ingest_file now
from app.qa import answer_question
from app.vector_store import list_documents

app = FastAPI(title="AI Knowledge Search Engine")

# CORS – React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
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
async def ask_question(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer, chunks = answer_question(
        payload.question,
        k=payload.top_k,
        doc_name=payload.doc_name,
    )
    return AskResponse(answer=answer, context=chunks)


@app.get("/documents")
async def get_documents():
    return {"documents": list_documents()}
