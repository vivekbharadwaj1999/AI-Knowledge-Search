import json
import math
import os
from typing import Any, Dict, List, Optional

VECTOR_STORE_PATH = os.path.join("data", "vector_store.jsonl")


def _ensure_dir():
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)


def add_embeddings(texts: List[str], embeddings: List[List[float]], doc_name: str) -> None:
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


def _dot(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _neg_l2(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return -1e9
    return -math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def _neg_l1(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return -1e9
    return -sum(abs(x - y) for x, y in zip(a, b))


def _keyword_overlap_score(query_text: str, text: str) -> float:
    q_tokens = {t for t in query_text.lower().split() if t}
    d_tokens = {t for t in text.lower().split() if t}
    if not q_tokens or not d_tokens:
        return 0.0
    inter = len(q_tokens & d_tokens)
    union = len(q_tokens | d_tokens)
    return inter / union


def similarity_search(
    query_embedding: List[float],
    k: int = 5,
    doc_name: Optional[str] = None,
    similarity: str = "cosine",
    query_text: Optional[str] = None,
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

    if doc_name is not None:
        records = [r for r in records if r.get("doc_name") == doc_name]
        if not records:
            return []

    results: List[Dict[str, Any]] = []
    for rec in records:
        emb = rec.get("embedding")
        if not isinstance(emb, list):
            continue

        text = rec.get("text") or ""

        if similarity == "dot":
            score = _dot(query_embedding, emb)
        elif similarity == "neg_l2":
            score = _neg_l2(query_embedding, emb)
        elif similarity == "neg_l1":
            score = _neg_l1(query_embedding, emb)
        elif similarity == "hybrid" and query_text:
            base = _cosine_similarity(query_embedding, emb)
            kw = _keyword_overlap_score(query_text, text)
            score = 0.7 * base + 0.3 * kw
        else:
            score = _cosine_similarity(query_embedding, emb)

        enriched = dict(rec)
        enriched["score"] = float(score)
        results.append(enriched)

    if not results:
        return []

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def get_document_text(doc_name: str, max_chars: int = 20000) -> str:
    records = _load_records()
    chunks: List[str] = []

    for rec in records:
        if rec.get("doc_name") == doc_name:
            text = rec.get("text") or ""
            if isinstance(text, str):
                chunks.append(text)

    if not chunks:
        raise ValueError(f"No chunks found for document: {doc_name}")

    full_text = "\n\n".join(chunks)
    if max_chars is not None and len(full_text) > max_chars:
        return full_text[:max_chars]
    return full_text


def list_documents() -> List[str]:
    names = set()
    for rec in _load_records():
        name = rec.get("doc_name")
        if name:
            names.add(name)
    return sorted(names)


def get_document_embeddings() -> Dict[str, List[float]]:
    records = _load_records()

    chunks_by_doc: Dict[str, List[dict]] = {}

    for rec in records:
        doc = rec.get("doc_name")
        emb = rec.get("embedding")
        text = rec.get("text")

        if not doc or not isinstance(emb, list):
            continue

        chunk_index = rec.get("chunk_index")
        if chunk_index is None:
            chunk_index = rec.get("chunk")
        if chunk_index is None:
            chunk_index = rec.get("index")
        if chunk_index is None:
            chunk_index = 99999

        chunks_by_doc.setdefault(doc, []).append({
            "idx": chunk_index,
            "embedding": emb
        })

    doc_vectors: Dict[str, List[float]] = {}

    for doc, chunks in chunks_by_doc.items():
        chunks.sort(key=lambda x: x["idx"])
        first = chunks[0]
        doc_vectors[doc] = first["embedding"]

    return doc_vectors


def get_document_previews(max_chars_per_doc: int = 1200) -> Dict[str, str]:
    records = _load_records()
    texts_by_doc: Dict[str, str] = {}

    for rec in records:
        doc = rec.get("doc_name")
        text = rec.get("text") or ""
        if not doc or not text:
            continue

        current = texts_by_doc.get(doc, "")
        if len(current) >= max_chars_per_doc:
            continue

        remaining = max_chars_per_doc - len(current)
        if remaining <= 0:
            continue

        addition = text[:remaining]
        if current:
            texts_by_doc[doc] = current + "\n\n" + addition
        else:
            texts_by_doc[doc] = addition

    return texts_by_doc


def clear_vector_store() -> None:
    if os.path.exists(VECTOR_STORE_PATH):
        os.remove(VECTOR_STORE_PATH)
