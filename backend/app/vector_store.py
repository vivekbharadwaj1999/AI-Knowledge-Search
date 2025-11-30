import json
import math
import os
from typing import Any, Dict, List, Optional

VECTOR_STORE_PATH = os.path.join("data", "vector_store.jsonl")


def _ensure_dir():
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)


def add_embeddings(texts: List[str], embeddings: List[List[float]], doc_name: str) -> None:
    """
    Append (text, embedding, doc_name) records to the JSONL vector store.
    """
    _ensure_dir()
    with open(VECTOR_STORE_PATH, "a", encoding="utf-8") as f:
        for text, emb in zip(texts, embeddings):
            rec: Dict[str, Any] = {
                "doc_name": doc_name,
                "text": text,
                "embedding": emb,
            }
            f.write(json.dumps(rec) + "\n")


def _load_records() -> List[Dict[str, Any]]:
    if not os.path.exists(VECTOR_STORE_PATH):
        return []
    records: List[Dict[str, Any]] = []
    with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def get_latest_doc_name() -> Optional[str]:
    """
    Return the doc_name of the most recently ingested document
    (i.e., the last record in the file).
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        return None

    last: Optional[Dict[str, Any]] = None
    with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue

    if last is None:
        return None
    return last.get("doc_name")


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def similarity_search(
    query_embedding: List[float],
    k: int = 5,
    doc_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return top-k records most similar to the query.

    Each record is a dict with at least:
      - "doc_name": str
      - "text": str
      - "embedding": List[float]
      - "score": float (similarity to the query)

    - If doc_name is None, search across ALL documents.
    - If doc_name is provided, restrict the search to that single document.
    """
    records = _load_records()
    if not records:
        return []

    # If a specific document is chosen, filter down to that doc only.
    if doc_name is not None:
        records = [r for r in records if r.get("doc_name") == doc_name]
        if not records:
            return []

    results: List[Dict[str, Any]] = []
    for rec in records:
        emb = rec.get("embedding")
        if not isinstance(emb, list):
            continue

        score = _cosine_similarity(query_embedding, emb)
        enriched = dict(rec)
        enriched["score"] = float(score)
        results.append(enriched)

    if not results:
        return []

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def list_documents() -> List[str]:
    names = set()
    for rec in _load_records():
        name = rec.get("doc_name")
        if name:
            names.add(name)
    return sorted(names)


def clear_vector_store() -> None:
    """
    Remove ALL stored vectors / documents by deleting the JSONL file.
    """
    if os.path.exists(VECTOR_STORE_PATH):
        os.remove(VECTOR_STORE_PATH)
