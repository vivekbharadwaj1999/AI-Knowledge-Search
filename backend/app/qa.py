import math
from typing import List, Optional, Dict, Any
from app.config import EmbeddingClient, LLMClient
from app.vector_store import similarity_search, _load_records
from app.critique import run_critique


def build_prompt(question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context_text}

Question: {question}

- First, try to infer the best answer you can from the context, even if it is not stated in a single sentence.
- If you truly cannot infer an answer at all, then say you don't know.
- Context chunks may indicate their source as "[Source: DOC_NAME]". If different sources disagree, briefly point this out and explain.
- Be clear and concise.
"""


def answer_question(
    question: str,
    k: int = 7,
    doc_name: Optional[str] = None,
    model: Optional[str] = None,
    similarity: Optional[str] = None,
):
    embed_client = EmbeddingClient()
    query_embedding = embed_client.embed_query(question)

    records = similarity_search(
        query_embedding,
        k=k,
        doc_name=doc_name,
        similarity=similarity or "cosine",
        query_text=question,
    )

    context_for_llm: List[str] = []
    sources: List[dict] = []

    for rec in records:
        text = rec.get("text") or ""
        if not text:
            continue

        doc = rec.get("doc_name") or "Unknown document"
        score = float(rec.get("score", 0.0))

        labeled = f"[Source: {doc}] {text}"
        context_for_llm.append(labeled)

        sources.append(
            {
                "doc_name": doc,
                "text": text,
                "score": score,
            }
        )

    llm = LLMClient()
    prompt = build_prompt(question, context_for_llm)
    answer = llm.complete(prompt, model=model)

    plain_chunks = [s["text"] for s in sources]

    return answer, plain_chunks, sources


def calculate_all_similarities(query_embedding: List[float], chunk_embedding: List[float], query_text: str, chunk_text: str) -> Dict[str, float]:
    """Calculate all 5 similarity metrics for a chunk"""

    def _cosine(a, b):
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def _dot(a, b):
        return float(sum(x * y for x, y in zip(a, b))) if len(a) == len(b) else 0.0

    def _neg_l2(a, b):
        if len(a) != len(b):
            return -1e9
        return -math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))

    def _neg_l1(a, b):
        if len(a) != len(b):
            return -1e9
        return -sum(abs(x - y) for x, y in zip(a, b))

    def _keyword_overlap(q_text, c_text):
        q_tokens = set(q_text.lower().split())
        c_tokens = set(c_text.lower().split())
        if not q_tokens or not c_tokens:
            return 0.0
        return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)

    cosine = _cosine(query_embedding, chunk_embedding)

    return {
        "cosine": cosine,
        "dot": _dot(query_embedding, chunk_embedding),
        "neg_l2": _neg_l2(query_embedding, chunk_embedding),
        "neg_l1": _neg_l1(query_embedding, chunk_embedding),
        "hybrid": 0.7 * cosine + 0.3 * _keyword_overlap(query_text, chunk_text)
    }


def get_chunks_for_all_methods(
    query_text: str,
    k: int = 7,
    doc_name: Optional[str] = None
) -> Dict[str, Any]:
    embed_client = EmbeddingClient()
    query_embedding = embed_client.embed_query(query_text)
    all_records = _load_records()
    if not all_records:
        return {"error": "No documents available"}

    if doc_name:
        all_records = [r for r in all_records if r.get("doc_name") == doc_name]
        if not all_records:
            return {"error": f"No chunks for document: {doc_name}"}

    chunks_with_scores = []
    for rec in all_records:
        emb = rec.get("embedding")
        if not isinstance(emb, list):
            continue

        text = rec.get("text", "")
        scores = calculate_all_similarities(
            query_embedding, emb, query_text, text)

        chunks_with_scores.append({
            "doc_name": rec.get("doc_name", "Unknown"),
            "text": text,
            "all_scores": scores,
            "chunk_length": len(text)
        })

    top_k_by_method = {}
    for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
        sorted_chunks = sorted(
            chunks_with_scores,
            key=lambda c: c["all_scores"][method],
            reverse=True
        )

        top_k_by_method[method] = [
            {
                **chunk,
                "rank": i + 1,
                "primary_score": chunk["all_scores"][method]
            }
            for i, chunk in enumerate(sorted_chunks[:k])
        ]

    similarity_stats = {}
    for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
        scores = [c["all_scores"][method] for c in chunks_with_scores]
        similarity_stats[method] = {
            "min": float(min(scores)),
            "max": float(max(scores)),
            "avg": float(sum(scores) / len(scores)),
        }

    methods = ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]
    method_agreement = {}
    for m1 in methods:
        method_agreement[m1] = {}
        chunks_m1 = {c["text"] for c in top_k_by_method[m1]}
        for m2 in methods:
            chunks_m2 = {c["text"] for c in top_k_by_method[m2]}
            overlap = len(chunks_m1 & chunks_m2)
            method_agreement[m1][m2] = round(
                (overlap / k) * 100, 1) if k > 0 else 0

    return {
        "query_embedding": query_embedding,
        "all_chunks_with_scores": chunks_with_scores,
        "top_k_by_method": top_k_by_method,
        "retrieval_details": {
            "query_analysis": {
                "original_query": query_text,
                "query_length": len(query_text),
                "query_tokens": len(query_text.split()),
            },
            "retrieval_config": {
                "top_k": k,
                "doc_filter": doc_name,
                "total_chunks_available": len(all_records),
            },
            "similarity_stats": similarity_stats,
            "method_agreement": method_agreement,
        }
    }


def analyze_ask_with_all_methods(
    question: str,
    k: int = 7,
    doc_name: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    retrieval_data = get_chunks_for_all_methods(question, k, doc_name)
    if "error" in retrieval_data:
        return retrieval_data

    llm = LLMClient()
    results_by_method = {}

    for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
        chunks = retrieval_data["top_k_by_method"][method]

        context_chunks = []
        for chunk in chunks:
            labeled = f"[Source: {chunk['doc_name']}] {chunk['text']}"
            context_chunks.append(labeled)

        from app.qa import build_prompt
        prompt = build_prompt(question, context_chunks)

        answer = llm.complete(prompt, model=model)

        results_by_method[method] = {
            "sources": [
                {
                    "rank": c["rank"],
                    "doc_name": c["doc_name"],
                    "text": c["text"],
                    "text_preview": c["text"][:80] + "..." if len(c["text"]) > 80 else c["text"],
                    "score": c["primary_score"],
                    "all_scores": c["all_scores"]
                }
                for c in chunks
            ],
            "answer": answer,
            "answer_length": len(answer),
            "context_used": len(context_chunks)
        }

    return {
        "input": {
            "question": question,
            "model": model or "default",
            "top_k": k
        },
        "retrieval_details": retrieval_data["retrieval_details"],
        "results_by_method": results_by_method
    }


def analyze_compare_with_all_methods(
    question: str,
    models: List[str],
    k: int = 7,
    doc_name: Optional[str] = None
) -> Dict[str, Any]:
    retrieval_data = get_chunks_for_all_methods(question, k, doc_name)
    if "error" in retrieval_data:
        return retrieval_data

    llm = LLMClient()
    results_by_method = {}

    for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
        chunks = retrieval_data["top_k_by_method"][method]

        context_chunks = []
        for chunk in chunks:
            labeled = f"[Source: {chunk['doc_name']}] {chunk['text']}"
            context_chunks.append(labeled)

        from app.qa import build_prompt
        prompt = build_prompt(question, context_chunks)

        answers_by_model = {}
        for model in models:
            answer = llm.complete(prompt, model=model)
            answers_by_model[model] = {
                "answer": answer,
                "length": len(answer)
            }

        results_by_method[method] = {
            "sources": [
                {
                    "rank": c["rank"],
                    "doc_name": c["doc_name"],
                    "text": c["text"],
                    "text_preview": c["text"][:80] + "..." if len(c["text"]) > 80 else c["text"],
                    "score": c["primary_score"],
                    "all_scores": c["all_scores"],
                }
                for c in chunks
            ],
            "answers_by_model": answers_by_model,
            "context_used": len(context_chunks)
        }

    return {
        "operation": "compare",
        "input": {
            "question": question,
            "models": models,
            "top_k": k
        },
        "retrieval_details": retrieval_data["retrieval_details"],
        "results_by_method": results_by_method
    }


def analyze_critique_with_all_methods(
    question: str,
    answer_model: str,
    critic_model: str,
    k: int = 7,
    doc_name: Optional[str] = None,
    self_correct: bool = True,
) -> Dict[str, Any]:
    retrieval_data = get_chunks_for_all_methods(question, k, doc_name)
    if "error" in retrieval_data:
        return retrieval_data

    results_by_method = {}

    for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
        critique_result = run_critique(
            question=question,
            answer_model=answer_model,
            critic_model=critic_model,
            top_k=k,
            doc_name=doc_name,
            self_correct=self_correct,
            similarity=method,
        )

        chunks = retrieval_data["top_k_by_method"][method]

        results_by_method[method] = {
            "sources": [
                {
                    "rank": c["rank"],
                    "doc_name": c["doc_name"],
                    "text": c["text"], 
                    "text_preview": c["text"][:80] + "..." if len(c["text"]) > 80 else c["text"],
                    "score": c["primary_score"],
                    "all_scores": c["all_scores"], 
                }
                for c in chunks
            ],
            "critique_result": critique_result,
            "context_used": len(chunks),
        }

    return {
        "operation": "critique", 
        "input": {
            "question": question,
            "answer_model": answer_model,
            "critic_model": critic_model,
            "top_k": k,
            "self_correct": self_correct,
        },
        "retrieval_details": retrieval_data["retrieval_details"],
        "results_by_method": results_by_method,
    }

