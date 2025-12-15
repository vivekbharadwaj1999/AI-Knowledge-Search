import math
from typing import List, Optional, Dict, Any
from app.config import EmbeddingClient, LLMClient
from app.vector_store import similarity_search, _load_records, get_document_embedding_model
from app.critique import run_critique


def _l2_normalize(v: List[float], eps: float = 1e-12) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n <= eps:
        return v
    return [x / n for x in v]


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
    normalize_vectors: bool = True,
    embedding_model: Optional[str] = None,
):
    """
    Answer a question using the vector store.
    
    Args:
        question: The question to answer
        k: Number of top chunks to retrieve
        doc_name: Optional document name to filter by
        model: LLM model to use for answering
        similarity: Similarity metric to use
        normalize_vectors: Whether to normalize vectors
        embedding_model: Embedding model to use. If None and doc_name is provided,
                        uses the model that was used to embed that document.
    """
    # Determine which embedding model to use
    if embedding_model is None and doc_name is not None:
        # Try to get the embedding model used for this document
        embedding_model = get_document_embedding_model(doc_name)
    
    embed_client = EmbeddingClient(model_name=embedding_model)
    query_embedding = embed_client.embed_query(question)

    records = similarity_search(
        query_embedding,
        k=k,
        doc_name=doc_name,
        similarity=similarity or "cosine",
        query_text=question,
        normalize_vectors=normalize_vectors,
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


def calculate_all_similarities(
    query_embedding: List[float],
    chunk_embedding: List[float],
    chunk_text: str,
    query_text: str,
    normalize_vectors: bool = True,
) -> Dict[str, float]:
    if normalize_vectors:
        query_embedding = _l2_normalize(query_embedding)
        chunk_embedding = _l2_normalize(chunk_embedding)

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
    doc_name: Optional[str] = None,
    normalize_vectors: bool = True,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get chunks using all similarity methods.
    
    Args:
        query_text: The query text
        k: Number of top chunks to retrieve
        doc_name: Optional document name to filter by
        normalize_vectors: Whether to normalize vectors
        embedding_model: Embedding model to use. If None and doc_name is provided,
                        uses the model that was used to embed that document.
    """
    # Determine which embedding model to use
    if embedding_model is None and doc_name is not None:
        embedding_model = get_document_embedding_model(doc_name)
    
    embed_client = EmbeddingClient(model_name=embedding_model)
    query_embedding = embed_client.embed_query(query_text)
    embedding_dimension = len(query_embedding)
    embedding_preview = query_embedding[:100] if embedding_dimension > 0 else []
    
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
            query_embedding=query_embedding,
            chunk_embedding=emb,
            chunk_text=text,
            query_text=query_text,
            normalize_vectors=normalize_vectors,
        )

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
                "embedding_dimension": embedding_dimension,
                "embedding_preview": embedding_preview,
                "embedding_model": embedding_model,
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
    model: Optional[str] = None,
    normalize_vectors: bool = True,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    retrieval_data = get_chunks_for_all_methods(
        question, k, doc_name, normalize_vectors=normalize_vectors,
        embedding_model=embedding_model
    )
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
        "operation": "ask",
        "input": {
            "question": question,
            "model": model or "default",
            "top_k": k,
            "embedding_model": embedding_model,
        },
        "retrieval_details": retrieval_data["retrieval_details"],
        "results_by_method": results_by_method,
        "query_embedding": retrieval_data.get("query_embedding"),
    }


def analyze_compare_with_all_methods(
    question: str,
    models: List[str],
    k: int = 7,
    doc_name: Optional[str] = None,
    normalize_vectors: bool = True,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    retrieval_data = get_chunks_for_all_methods(
        question, k, doc_name, normalize_vectors=normalize_vectors,
        embedding_model=embedding_model
    )
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
            "top_k": k,
            "embedding_model": embedding_model,
        },
        "retrieval_details": retrieval_data["retrieval_details"],
        "results_by_method": results_by_method,
        "query_embedding": retrieval_data.get("query_embedding"),
    }


def analyze_critique_with_all_methods(
    question: str,
    answer_model: str,
    critic_model: str,
    k: int = 7,
    doc_name: Optional[str] = None,
    self_correct: bool = True,
    normalize_vectors: bool = True,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    retrieval_data = get_chunks_for_all_methods(
        question, k, doc_name, normalize_vectors=normalize_vectors,
        embedding_model=embedding_model
    )
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
            embedding_model=embedding_model,
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
            "embedding_model": embedding_model,
        },
        "retrieval_details": retrieval_data["retrieval_details"],
        "results_by_method": results_by_method,
        "query_embedding": retrieval_data.get("query_embedding"),
    }
