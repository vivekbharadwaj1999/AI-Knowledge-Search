import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import json
from pathlib import Path
from datetime import datetime

from app.schemas import (
    AskRequest,
    AskResponse,
    CompareRequest,
    CompareResponse,
    CompareResult,
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
    SignupRequest,
    LoginRequest,
    AuthResponse,
    UserResponse,
)
from app.ingest import ingest_file, UPLOAD_DIR
from app.qa import answer_question
from app.vector_store import list_documents, clear_vector_store
from app.config import GROQ_MODEL, AVAILABLE_EMBEDDING_MODELS, get_embedding_dimension
from app.insights import generate_insights
from app.report import generate_document_report
from app.relations import analyze_cross_document_relations
from app.critique import (
    run_critique,
    reset_critique_log_file,
)
from app.auth import (
    create_user,
    authenticate_user,
    create_guest_session,
    decode_token,
    get_user_upload_dir,
    get_user_critique_log_path,
    cleanup_guest_data,
    delete_user_account,
)
from app.operations_log import (
    log_ask_operation,
    log_compare_operation,
    log_advanced_analysis_operation,
    log_critique_operation,
    get_operations_log,
    reset_operations_log,
    check_operations_log_exists,
)
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Knowledge Search Engine")

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://46.224.82.38",
    "http://46.224.82.38:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)


async def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[Dict]:
    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        return None

    token = authorization[7:] 
    user_data = decode_token(token)
    return user_data


@app.get("/")
def root():
    return {"status": "ok", "message": "AI Knowledge Search backend running"}


@app.post("/auth/signup", response_model=AuthResponse)
async def signup(payload: SignupRequest):
    try:
        success = create_user(payload.username, payload.password)
        if not success:
            raise HTTPException(
                status_code=400, detail="Username already exists")

        token = authenticate_user(payload.username, payload.password)
        return AuthResponse(token=token, username=payload.username, is_guest=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login", response_model=AuthResponse)
async def login(payload: LoginRequest):
    token = authenticate_user(payload.username, payload.password)
    if not token:
        raise HTTPException(
            status_code=401, detail="Invalid username or password")

    return AuthResponse(token=token, username=payload.username, is_guest=False)


@app.post("/auth/guest", response_model=AuthResponse)
async def create_guest():
    token = create_guest_session()
    user_data = decode_token(token)
    return AuthResponse(token=token, username=user_data["username"], is_guest=True)


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(authorization: Optional[str] = Header(None)):
    user_data = await get_current_user_optional(authorization)
    if user_data:
        return UserResponse(username=user_data["username"], is_guest=user_data["is_guest"])

    raise HTTPException(status_code=401, detail="Not authenticated")


@app.post("/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    user_data = await get_current_user_optional(authorization)
    if user_data and user_data.get("is_guest"):
        cleanup_guest_data(user_data["username"])

    return {"status": "ok", "message": "Logged out"}


@app.delete("/auth/account")
async def delete_account(authorization: Optional[str] = Header(None)):
    user_data = await get_current_user_optional(authorization)

    if not user_data:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if user_data.get("is_guest"):
        raise HTTPException(
            status_code=400, detail="Guest users cannot delete accounts")

    username = user_data["username"]

    success = delete_user_account(username)

    if not success:
        raise HTTPException(status_code=404, detail="User not found")

    return {"status": "ok", "message": f"Account '{username}' deleted successfully"}


@app.get("/embedding-models")
async def get_embedding_models():
    models = []
    for model_id, info in AVAILABLE_EMBEDDING_MODELS.items():
        models.append({
            "id": model_id,
            "label": info["label"],
            "type": info["type"],
            "dimension": info["dimension"],
            "description": info["description"]
        })
    return {"models": models}


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(200),
    embedding_model: str = Form("all-MiniLM-L6-v2"),
    authorization: Optional[str] = Header(None),
):
    user_info = await get_current_user_optional(authorization)
    username = user_info["username"] if user_info else "default"
    is_guest = user_info.get("is_guest", True) if user_info else True

    allowed_exts = (".pdf", ".txt", ".csv", ".docx", ".pptx", ".xlsx")
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: PDF, TXT, CSV, DOCX, PPTX, XLSX.",
        )

    if embedding_model not in AVAILABLE_EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid embedding model. Choose from: {', '.join(AVAILABLE_EMBEDDING_MODELS.keys())}"
        )

    user_upload_dir = get_user_upload_dir(username, is_guest)
    os.makedirs(user_upload_dir, exist_ok=True)

    file_path = os.path.join(user_upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        chunk_count = ingest_file(
            file_path,
            doc_name=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            username=username,
            is_guest=is_guest,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    if chunk_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in file",
        )

    embedding_dim = get_embedding_dimension(embedding_model)

    return {
        "status": "ok",
        "chunks_indexed": chunk_count,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dim,
        "is_guest": is_guest
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question_route(payload: AskRequest, authorization: Optional[str] = Header(None)):
    user_info = await get_current_user_optional(authorization)
    username = user_info["username"] if user_info else "default"
    is_guest = user_info.get("is_guest", True) if user_info else True

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    embedding_model = getattr(payload, 'embedding_model', None)
    if embedding_model and embedding_model not in AVAILABLE_EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid embedding model: {embedding_model}"
        )

    answer, chunks, sources = answer_question(
        payload.question,
        k=payload.top_k,
        doc_name=payload.doc_name,
        model=payload.model,
        similarity=payload.similarity,
        normalize_vectors=payload.normalize_vectors,
        embedding_model=embedding_model,
        username=username,
        is_guest=is_guest,
    )

    model_used = payload.model or GROQ_MODEL

    log_ask_operation(
        question=payload.question,
        answer=answer,
        context=chunks,
        sources=sources,
        top_k=payload.top_k,
        doc_name=payload.doc_name,
        model=model_used,
        similarity=payload.similarity or "cosine",
        normalize_vectors=payload.normalize_vectors,
        embedding_model=embedding_model,
        temperature=None,
        username=username,
        is_guest=is_guest,
    )

    return AskResponse(
        answer=answer,
        context=chunks,
        model_used=model_used,
        sources=sources,
    )


@app.post("/compare", response_model=CompareResponse)
async def compare_route(payload: CompareRequest, authorization: Optional[str] = Header(None)):
    user_info = await get_current_user_optional(authorization)
    username = user_info["username"] if user_info else "default"
    is_guest = user_info.get("is_guest", True) if user_info else True

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    embedding_model = getattr(payload, 'embedding_model', None)
    if embedding_model and embedding_model not in AVAILABLE_EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid embedding model: {embedding_model}"
        )

    answer_left, chunks_left, sources_left = answer_question(
        payload.question,
        k=payload.top_k,
        doc_name=payload.doc_name,
        model=payload.model_left,
        similarity=payload.similarity,
        normalize_vectors=payload.normalize_vectors,
        embedding_model=embedding_model,
        username=username,
        is_guest=is_guest,
    )

    answer_right, chunks_right, sources_right = answer_question(
        payload.question,
        k=payload.top_k,
        doc_name=payload.doc_name,
        model=payload.model_right,
        similarity=payload.similarity,
        normalize_vectors=payload.normalize_vectors,
        embedding_model=embedding_model,
        username=username,
        is_guest=is_guest,
    )

    log_compare_operation(
        question=payload.question,
        model_left=payload.model_left,
        model_right=payload.model_right,
        answer_left=answer_left,
        answer_right=answer_right,
        context_left=chunks_left,
        context_right=chunks_right,
        sources_left=sources_left,
        sources_right=sources_right,
        top_k=payload.top_k,
        doc_name=payload.doc_name,
        similarity=payload.similarity or "cosine",
        normalize_vectors=payload.normalize_vectors,
        embedding_model=embedding_model,
        temperature=None,
        username=username,
        is_guest=is_guest,
    )

    return CompareResponse(
        left=CompareResult(
            model=payload.model_left,
            answer=answer_left,
            context=chunks_left,
            sources=sources_left,
        ),
        right=CompareResult(
            model=payload.model_right,
            answer=answer_right,
            context=chunks_right,
            sources=sources_right,
        ),
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
async def get_documents(authorization: Optional[str] = Header(None)):
    user_info = await get_current_user_optional(authorization)
    username = user_info["username"] if user_info else "default"
    is_guest = user_info.get("is_guest", True) if user_info else True
    return {"documents": list_documents(username, is_guest)}


@app.post("/report", response_model=DocumentReport)
async def create_report(req: ReportRequest, authorization: Optional[str] = Header(None)):
    user_info = await get_current_user_optional(authorization)
    if user_info:
        username, is_guest = user_info["username"], user_info.get(
            "is_guest", True)
    else:
        username, is_guest = "default", True

    try:
        report = generate_document_report(
            doc_name=req.doc_name,
            model=req.model,
            username=username,
            is_guest=is_guest,
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
async def document_relations_route(payload: CrossDocRelationsRequest, authorization: Optional[str] = Header(None)):
    user_info = await get_current_user_optional(authorization)
    username = user_info["username"] if user_info else "default"
    is_guest = user_info.get("is_guest", True) if user_info else True
    try:
        result = analyze_cross_document_relations(
            model=payload.model,
            max_pairs=payload.max_pairs,
            min_similarity=payload.min_similarity,
            similarity=payload.similarity,
            normalize_vectors=payload.normalize_vectors,
            username=username,
            is_guest=is_guest,
        )
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
async def critique_route(payload: CritiqueRequest, authorization: Optional[str] = Header(None)):
    user_info = await get_current_user_optional(authorization)
    username = user_info["username"] if user_info else "default"
    is_guest = user_info.get("is_guest", True) if user_info else True

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
        normalize_vectors=payload.normalize_vectors,
        embedding_model=payload.embedding_model,
        username=username,
        is_guest=is_guest,
    )

    log_critique_operation(
        question=payload.question,
        answer_model=payload.answer_model,
        critic_model=data["critic_model"],
        answer=data["answer"],
        context=data["context"],
        sources=data.get("sources", []),
        answer_critique_markdown=data["answer_critique_markdown"],
        prompt_feedback_markdown=data["prompt_feedback_markdown"],
        improved_prompt=data["improved_prompt"],
        prompt_issue_tags=data["prompt_issue_tags"],
        scores=data.get("scores"),
        rounds=data["rounds"],
        top_k=payload.top_k,
        doc_name=payload.doc_name,
        self_correct=payload.self_correct,
        similarity=payload.similarity or "cosine",
        normalize_vectors=payload.normalize_vectors,
        embedding_model=payload.embedding_model,
        temperature=None,
        username=username,
        is_guest=is_guest,
    )
    
    return data


@app.get("/critique-log-rows", response_model=CritiqueLogResponse)
async def get_critique_log_rows(authorization: Optional[str] = Header(None)):
    user_info = await get_current_user_optional(authorization)
    if user_info:
        username, is_guest = user_info["username"], user_info.get(
            "is_guest", True)
    else:
        username, is_guest = "default", True

    log_path = Path(get_user_critique_log_path(username, is_guest))
    if not log_path.exists():
        return CritiqueLogResponse(rows=[])

    rows: list[CritiqueLogRow] = []

    with log_path.open("r", encoding="utf-8") as f:
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


@app.get("/critique-log-exists")
async def check_critique_log_exists(authorization: Optional[str] = Header(None)):
    user_info = await get_current_user_optional(authorization)
    if user_info:
        username, is_guest = user_info["username"], user_info.get(
            "is_guest", True)
    else:
        username, is_guest = "default", True

    log_path = Path(get_user_critique_log_path(username, is_guest))

    if not log_path.exists():
        return {"exists": False, "count": 0}

    count = 0
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        count += 1
                    except:
                        pass
    except:
        return {"exists": False, "count": 0}

    return {"exists": count > 0, "count": count}


@app.post("/analyze")
async def analyze_operation(payload: dict, authorization: Optional[str] = Header(None)):
    user_info = await get_current_user_optional(authorization)
    username = user_info["username"] if user_info else "default"
    is_guest = user_info.get("is_guest", True) if user_info else True

    operation = payload.get("operation", "").lower()
    normalize_vectors = bool(payload.get("normalize_vectors", True))
    embedding_model = payload.get("embedding_model")
    temperature = payload.get("temperature")
    if temperature is not None:
        temperature = float(temperature)

    if operation not in ["ask", "compare", "critique"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid operation. Must be 'ask', 'compare', or 'critique'"
        )

    if embedding_model and embedding_model not in AVAILABLE_EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid embedding model: {embedding_model}"
        )

    from app.qa import (
        analyze_ask_with_all_methods,
        analyze_compare_with_all_methods,
        analyze_critique_with_all_methods
    )

    if operation == "ask":
        question = payload.get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question required")

        result = analyze_ask_with_all_methods(
            question=question,
            k=payload.get("top_k", 7),
            doc_name=payload.get("doc_name"),
            model=payload.get("model"),
            normalize_vectors=payload.get("normalize_vectors", True),
            embedding_model=embedding_model,
            temperature=temperature,
            username=username,
            is_guest=is_guest,
        )

        log_advanced_analysis_operation(
            operation="ask",
            parameters={
                "question": question,
                "top_k": payload.get("top_k", 7),
                "doc_name": payload.get("doc_name"),
                "model": payload.get("model"),
                "normalize_vectors": payload.get("normalize_vectors", True),
                "embedding_model": embedding_model,
                "temperature": temperature,
            },
            results=result,
            username=username,
            is_guest=is_guest,
        )

    elif operation == "compare":
        question = payload.get("question", "").strip()
        models = payload.get("models", [])

        if not question:
            raise HTTPException(status_code=400, detail="Question required")
        if not models or len(models) < 2:
            raise HTTPException(
                status_code=400, detail="Need at least 2 models")

        result = analyze_compare_with_all_methods(
            question=question,
            models=models,
            k=payload.get("top_k", 7),
            doc_name=payload.get("doc_name"),
            normalize_vectors=payload.get("normalize_vectors", True),
            embedding_model=embedding_model,
            temperature=temperature,
            username=username,
            is_guest=is_guest,
        )

        log_advanced_analysis_operation(
            operation="compare",
            parameters={
                "question": question,
                "models": models,
                "top_k": payload.get("top_k", 7),
                "doc_name": payload.get("doc_name"),
                "normalize_vectors": payload.get("normalize_vectors", True),
                "embedding_model": embedding_model,
                "temperature": temperature,
            },
            results=result,
            username=username,
            is_guest=is_guest,
        )

    elif operation == "critique":
        question = payload.get("question", "").strip()
        answer_model = payload.get("answer_model")
        critic_model = payload.get("critic_model")
        max_rounds = int(payload.get("max_rounds", 2))
        self_correct = max_rounds > 1

        if not question:
            raise HTTPException(status_code=400, detail="Question required")
        if not answer_model:
            raise HTTPException(
                status_code=400, detail="answer_model required")
        if not critic_model:
            raise HTTPException(
                status_code=400, detail="critic_model required")

        result = analyze_critique_with_all_methods(
            question=question,
            answer_model=answer_model,
            critic_model=critic_model,
            k=payload.get("top_k", 7),
            doc_name=payload.get("doc_name"),
            self_correct=self_correct,
            normalize_vectors=payload.get("normalize_vectors", True),
            embedding_model=embedding_model,
            temperature=temperature,
            username=username,
            is_guest=is_guest,
        )

        log_advanced_analysis_operation(
            operation="critique",
            parameters={
                "question": question,
                "answer_model": answer_model,
                "critic_model": critic_model,
                "top_k": payload.get("top_k", 7),
                "doc_name": payload.get("doc_name"),
                "self_correct": self_correct,
                "normalize_vectors": payload.get("normalize_vectors", True),
                "embedding_model": embedding_model,
                "temperature": temperature,
            },
            results=result,
            username=username,
            is_guest=is_guest,
        )

    return {
        "operation": operation,
        **result
    }


@app.post("/reset-critique-log")
async def reset_critique_log_endpoint(authorization: Optional[str] = Header(None)):
    user_info = await get_current_user_optional(authorization)
    if user_info:
        username, is_guest = user_info["username"], user_info.get(
            "is_guest", True)
    else:
        username, is_guest = "default", True
    reset_critique_log_file(username, is_guest)
    return {"status": "ok", "message": "Critique log reset"}


@app.get("/operations-log")
async def get_operations_log_endpoint(authorization: Optional[str] = Header(None)):
    """Get all operations log entries for the current user."""
    user_info = await get_current_user_optional(authorization)
    if user_info:
        username, is_guest = user_info["username"], user_info.get("is_guest", True)
    else:
        username, is_guest = "default", True
    
    entries = get_operations_log(username, is_guest)
    return {"entries": entries, "count": len(entries)}


@app.get("/operations-log-exists")
async def check_operations_log_exists_endpoint(authorization: Optional[str] = Header(None)):
    """Check if operations log exists and has entries."""
    user_info = await get_current_user_optional(authorization)
    if user_info:
        username, is_guest = user_info["username"], user_info.get("is_guest", True)
    else:
        username, is_guest = "default", True
    
    exists = check_operations_log_exists(username, is_guest)
    
    if exists:
        entries = get_operations_log(username, is_guest)
        return {"exists": True, "count": len(entries)}
    
    return {"exists": False, "count": 0}


@app.post("/reset-operations-log")
async def reset_operations_log_endpoint(authorization: Optional[str] = Header(None)):
    """Reset (delete) the operations log for the current user."""
    user_info = await get_current_user_optional(authorization)
    if user_info:
        username, is_guest = user_info["username"], user_info.get("is_guest", True)
    else:
        username, is_guest = "default", True
    
    reset_operations_log(username, is_guest)
    return {"status": "ok", "message": "Operations log reset"}


@app.delete("/documents")
async def delete_all_documents(authorization: Optional[str] = Header(None)):
    user_info = await get_current_user_optional(authorization)
    if user_info:
        username, is_guest = user_info["username"], user_info.get(
            "is_guest", True)
    else:
        username, is_guest = "default", True

    user_upload_dir = get_user_upload_dir(username, is_guest)
    if os.path.exists(user_upload_dir):
        for fname in os.listdir(user_upload_dir):
            fpath = os.path.join(user_upload_dir, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except Exception as e:
                    print("Delete failed:", e)

    clear_vector_store(username, is_guest)

    return {"status": "ok", "message": "All documents removed"}


@app.post("/batch-evaluate")
async def run_batch_evaluation(payload: dict, authorization: Optional[str] = Header(None)):
    """Run batch evaluation across multiple configurations."""
    from app.batch_evaluation import BatchEvaluator
    
    user_info = await get_current_user_optional(authorization)
    username = user_info["username"] if user_info else "default"
    is_guest = user_info.get("is_guest", True) if user_info else True
    questions = payload.get("questions", [])
    operations = payload.get("operations", [])
    similarity_methods = payload.get("similarity_methods")
    embedding_models = payload.get("embedding_models")
    top_k_values = payload.get("top_k_values")
    doc_name = payload.get("doc_name")
    normalize_vectors = payload.get("normalize_vectors", True)
    temperature = payload.get("temperature")
    include_faithfulness = payload.get("include_faithfulness", True)

    if not questions:
        raise HTTPException(status_code=400, detail="Questions are required")
    
    if not operations:
        raise HTTPException(status_code=400, detail="At least one operation is required")

    evaluator = BatchEvaluator(username=username, is_guest=is_guest)
    
    results = evaluator.run_batch_experiment(
        questions=questions,
        operations=operations,
        similarity_methods=similarity_methods,
        embedding_models=embedding_models,
        top_k_values=top_k_values,
        doc_name=doc_name,
        normalize_vectors=normalize_vectors,
        temperature=temperature,
        include_faithfulness=include_faithfulness
    )
    
    return results


@app.post("/batch-evaluate/export")
async def export_batch_results(payload: dict, authorization: Optional[str] = Header(None)):
    """Export batch evaluation results to JSON."""
    from app.batch_evaluation import BatchEvaluator
    
    user_info = await get_current_user_optional(authorization)
    username = user_info["username"] if user_info else "default"
    is_guest = user_info.get("is_guest", True) if user_info else True
    
    results_data = payload.get("results")
    
    evaluator = BatchEvaluator(username=username, is_guest=is_guest)
    user_dir = get_user_upload_dir(username, is_guest)
    output_dir = Path(user_dir) / "batch_exports"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"batch_eval_{timestamp}.json"
    output_path = output_dir / filename
    evaluator.export_to_json(results_data, str(output_path))
    
    return {
        "status": "success",
        "filename": filename,
        "path": str(output_path),
        "format": "json"
    }


@app.post("/counterfactual-analysis")
async def run_counterfactual(payload: dict, authorization: Optional[str] = Header(None)):
    from app.extended_analysis import run_counterfactual_analysis
    
    user_info = await get_current_user_optional(authorization)
    username = user_info["username"] if user_info else "default"
    is_guest = user_info.get("is_guest", True) if user_info else True
    
    question = payload.get("question")
    original_chunks = payload.get("original_chunks", [])
    counterfactual_type = payload.get("counterfactual_type", "remove_top")
    original_answer = payload.get("original_answer")
    
    result = run_counterfactual_analysis(
        question=question,
        original_chunks=original_chunks,
        counterfactual_type=counterfactual_type,
        k=payload.get("top_k", 7),
        doc_name=payload.get("doc_name"),
        model=payload.get("model"),
        similarity=payload.get("similarity", "cosine"),
        embedding_model=payload.get("embedding_model"),
        temperature=payload.get("temperature"),
        username=username,
        is_guest=is_guest,
        original_answer=original_answer 
    )
    
    return result

@app.get("/debug/documents-metadata")
async def debug_documents_metadata(authorization: Optional[str] = Header(None)):
    from app.vector_store import _load_records
    
    user_info = await get_current_user_optional(authorization)
    username = user_info["username"] if user_info else "default"
    is_guest = user_info.get("is_guest", True) if user_info else True
    records = _load_records(username, is_guest)
    docs = {}
    for rec in records:
        doc_name = rec.get("doc_name")
        if doc_name not in docs:
            docs[doc_name] = {
                "embedding_model": rec.get("embedding_model", "NOT_SET"),
                "chunks": 0,
                "sample_chunk": rec.get("text", "")[:100] + "..."
            }
        docs[doc_name]["chunks"] += 1
    
    return {
        "username": username,
        "is_guest": is_guest,
        "total_chunks": len(records),
        "documents": docs
    }

