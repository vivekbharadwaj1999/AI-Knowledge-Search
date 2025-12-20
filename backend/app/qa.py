import math
from typing import List, Optional, Dict, Any
from app.config import EmbeddingClient, LLMClient
from app.vector_store import similarity_search, _load_records, get_document_embedding_model, get_documents_info
from app.critique import run_critique
from rouge_score import rouge_scorer
from app.faithfulness import (
    calculate_faithfulness_metrics,
    calculate_retrieval_quality_metrics
)

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
    temperature: Optional[float] = None,
    username: Optional[str] = None,
    is_guest: bool = False,
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
        temperature: Temperature for LLM generation
    """
    if embedding_model is None:
        if doc_name is not None:
            embedding_model = get_document_embedding_model(doc_name, username=username, is_guest=is_guest)
        else:
            docs_info = get_documents_info(username=username, is_guest=is_guest)
            if docs_info:
                from collections import Counter
                models = [d.get('embedding_model') for d in docs_info if d.get('embedding_model')]
                if models:
                    embedding_model = Counter(models).most_common(1)[0][0]
    
    embed_client = EmbeddingClient(model_name=embedding_model)
    query_embedding = embed_client.embed_query(question)

    records = similarity_search(
        query_embedding,
        k=k,
        doc_name=doc_name,
        similarity=similarity or "cosine",
        query_text=question,
        normalize_vectors=normalize_vectors,
        username=username,
        is_guest=is_guest,
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
    answer = llm.complete(prompt, model=model, temperature=temperature)

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

def calculate_answer_stability(
    answers_by_method: Dict[str, str],
    selected_method: str,
    embedding_model: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate answer stability metrics comparing the selected method's answer
    to answers from other methods.
    
    Returns:
        Dict mapping each method to {"cosine_semantic": float, "rouge_l": float}
    """
    if selected_method not in answers_by_method:
        return {}
    
    selected_answer = answers_by_method[selected_method]
    
    embed_client = EmbeddingClient(model_name=embedding_model)
    selected_embedding = embed_client.embed_query(selected_answer)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    stability = {}
    
    for method, answer in answers_by_method.items():
        if method == selected_method:
            stability[method] = {
                "cosine_semantic": 1.0,
                "rouge_l": 1.0
            }
            continue
        
        other_embedding = embed_client.embed_query(answer)
        cosine_sim = sum(a * b for a, b in zip(selected_embedding, other_embedding))
        norm_selected = math.sqrt(sum(x * x for x in selected_embedding))
        norm_other = math.sqrt(sum(x * x for x in other_embedding))
        cosine_sim = cosine_sim / (norm_selected * norm_other) if norm_selected and norm_other else 0.0
        
        rouge_scores = scorer.score(selected_answer, answer)
        rouge_l_f1 = rouge_scores['rougeL'].fmeasure
        
        stability[method] = {
            "cosine_semantic": round(cosine_sim, 4),
            "rouge_l": round(rouge_l_f1, 4)
        }
    
    return stability

def get_chunks_for_all_methods(
    query_text: str,
    k: int = 7,
    doc_name: Optional[str] = None,
    normalize_vectors: bool = True,
    embedding_model: Optional[str] = None,
    username: Optional[str] = None,
    is_guest: bool = False,
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
    if embedding_model is None:
        if doc_name is not None:
            embedding_model = get_document_embedding_model(doc_name, username=username, is_guest=is_guest)
        else:
            docs_info = get_documents_info(username=username, is_guest=is_guest)
            if docs_info:
                from collections import Counter
                models = [d.get('embedding_model') for d in docs_info if d.get('embedding_model')]
                if models:
                    embedding_model = Counter(models).most_common(1)[0][0]
    
    embed_client = EmbeddingClient(model_name=embedding_model)
    query_embedding = embed_client.embed_query(query_text)
    embedding_dimension = len(query_embedding)
    embedding_preview = query_embedding[:100] if embedding_dimension > 0 else []
    
    all_records = _load_records(username=username, is_guest=is_guest)

    if all_records:
        if isinstance(all_records[0].get('embedding'), list):
            pass

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
    temperature: Optional[float] = None,
    username: Optional[str] = None,
    is_guest: bool = False,
) -> Dict[str, Any]:
    retrieval_data = get_chunks_for_all_methods(
        question, k, doc_name, normalize_vectors=normalize_vectors,
        embedding_model=embedding_model, username=username, is_guest=is_guest
    )
    if "error" in retrieval_data:
        return retrieval_data

    llm = LLMClient()
    results_by_method = {}
    answers_by_method = {}

    for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
        chunks = retrieval_data["top_k_by_method"][method]

        context_chunks = []
        for chunk in chunks:
            labeled = f"[Source: {chunk['doc_name']}] {chunk['text']}"
            context_chunks.append(labeled)

        from app.qa import build_prompt
        prompt = build_prompt(question, context_chunks)

        answer = llm.complete(prompt, model=model, temperature=temperature)
        answers_by_method[method] = answer

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

    # Add extended metrics (faithfulness and retrieval quality)
    for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
        answer = results_by_method[method]["answer"]
        chunks = results_by_method[method]["sources"]
        
        results_by_method[method]["extended_metrics"] = {
            "faithfulness": calculate_faithfulness_metrics(
                answer=answer,
                retrieved_chunks=chunks,
                question=question
            ),
            "retrieval_quality": calculate_retrieval_quality_metrics(
                retrieved_chunks=chunks,
                question=question
            )
        }

    answer_stability_by_method = {}
    for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
        answer_stability_by_method[method] = calculate_answer_stability(
            answers_by_method=answers_by_method,
            selected_method=method,
            embedding_model=embedding_model
        )

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
        "answer_stability": answer_stability_by_method,
    }

def analyze_compare_with_all_methods(
    question: str,
    models: List[str],
    k: int = 7,
    doc_name: Optional[str] = None,
    normalize_vectors: bool = True,
    embedding_model: Optional[str] = None,
    temperature: Optional[float] = None,
    username: Optional[str] = None,
    is_guest: bool = False,
) -> Dict[str, Any]:
    retrieval_data = get_chunks_for_all_methods(
        question, k, doc_name, normalize_vectors=normalize_vectors,
        embedding_model=embedding_model, username=username, is_guest=is_guest
    )
    if "error" in retrieval_data:
        return retrieval_data

    llm = LLMClient()
    results_by_method = {}

    answers_by_model = {model: {} for model in models}

    for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
        chunks = retrieval_data["top_k_by_method"][method]

        context_chunks = []
        for chunk in chunks:
            labeled = f"[Source: {chunk['doc_name']}] {chunk['text']}"
            context_chunks.append(labeled)

        from app.qa import build_prompt
        prompt = build_prompt(question, context_chunks)

        answers_by_model_for_method = {}
        for model in models:
            answer = llm.complete(prompt, model=model, temperature=temperature)
            answers_by_model_for_method[model] = {
                "answer": answer,
                "length": len(answer)
            }
            answers_by_model[model][method] = answer

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
            "answers_by_model": answers_by_model_for_method,
            "context_used": len(context_chunks)
        }

    # Add extended metrics for each method and model
    for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
        chunks = results_by_method[method]["sources"]
        results_by_method[method]["extended_metrics_by_model"] = {}
        
        for model in models:
            answer = results_by_method[method]["answers_by_model"][model]["answer"]
            results_by_method[method]["extended_metrics_by_model"][model] = {
                "faithfulness": calculate_faithfulness_metrics(
                    answer=answer,
                    retrieved_chunks=chunks,
                    question=question
                ),
                "retrieval_quality": calculate_retrieval_quality_metrics(
                    retrieved_chunks=chunks,
                    question=question
                )
            }

    answer_stability_by_model = {}
    for model in models:
        answer_stability_by_model[model] = {}
        for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
            answer_stability_by_model[model][method] = calculate_answer_stability(
                answers_by_method=answers_by_model[model],
                selected_method=method,
                embedding_model=embedding_model
            )

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
        "answer_stability": answer_stability_by_model,
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
    temperature: Optional[float] = None,
    username: Optional[str] = None,
    is_guest: bool = False,
) -> Dict[str, Any]:
    retrieval_data = get_chunks_for_all_methods(
        question, k, doc_name, normalize_vectors=normalize_vectors,
        embedding_model=embedding_model, username=username, is_guest=is_guest
    )
    if "error" in retrieval_data:
        return retrieval_data

    results_by_method = {}
    final_answers_by_method = {}

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
            temperature=temperature,
            username=username,
            is_guest=is_guest,
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

        final_answers_by_method[method] = critique_result.get("answer", "")

    # Add extended metrics
    for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
        answer = final_answers_by_method[method]
        chunks = results_by_method[method]["sources"]
        
        results_by_method[method]["extended_metrics"] = {
            "faithfulness": calculate_faithfulness_metrics(
                answer=answer,
                retrieved_chunks=chunks,
                question=question
            ),
            "retrieval_quality": calculate_retrieval_quality_metrics(
                retrieved_chunks=chunks,
                question=question
            )
        }

    answer_stability_by_method = {}
    for method in ["cosine", "dot", "neg_l2", "neg_l1", "hybrid"]:
        answer_stability_by_method[method] = calculate_answer_stability(
            answers_by_method=final_answers_by_method,
            selected_method=method,
            embedding_model=embedding_model
        )

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
        "answer_stability": answer_stability_by_method,
    }
