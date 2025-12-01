from typing import Any, Dict, List, Optional
import json

from app.config import LLMClient
from app.vector_store import get_document_embeddings, get_document_previews
from app.schemas import CrossDocRelations, DocPairRelation


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _safe_json_object(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    snippet = raw[start: end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return {}


def analyze_cross_document_relations(
    model: Optional[str] = None,
    max_pairs: int = 12,
    min_similarity: float = 0.2,
) -> CrossDocRelations:
    """
    Compute pairwise similarities between documents and let the LLM
    describe how they relate.
    """
    doc_embeddings = get_document_embeddings()
    doc_previews = get_document_previews()

    docs = list(doc_embeddings.keys())
    if len(docs) < 2:
        raise ValueError("Need at least two documents to analyze relations.")

    # 1) Build pair list
    pairs: List[Dict[str, Any]] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            a, b = docs[i], docs[j]
            sim = _cosine_similarity(doc_embeddings[a], doc_embeddings[b])
            pairs.append({"doc_a": a, "doc_b": b, "similarity": float(sim)})

    # sort & filter
    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    filtered_pairs = [p for p in pairs if p["similarity"] >= min_similarity]
    if not filtered_pairs:
        filtered_pairs = pairs[:max_pairs]
    else:
        filtered_pairs = filtered_pairs[:max_pairs]

    # 2) Build prompt for LLM
    doc_blocks = []
    for name in docs:
        preview = doc_previews.get(
            name, "").strip() or "(no preview available)"
        doc_blocks.append(f"DOC: {name}\nPREVIEW:\n{preview}\n---")

    pair_names = [
        f"{p['doc_a']} ↔ {p['doc_b']}"
        for p in filtered_pairs
    ]

    prompt = (
        "You are an expert academic reviewer. Your job is to explain, in depth, how documents relate "
        "based ONLY on their content.\n\n"

        "PROCESS (you MUST follow this step-by-step):\n"
        "1. For EACH DOCUMENT PREVIEW, infer its PRIMARY TOPIC in 1–3 words "
        "(e.g. 'performance modeling', 'IP traffic management', 'network analysis').\n"
        "2. For EACH PAIR of documents, compare ONLY their primary topics.\n"
        "3. If topics DO strongly overlap, produce a detailed academic relationship explanation.\n"
        "4. If topics DO NOT clearly overlap, describe the pair as conceptually more distant within the broader subject area.\n"
        "   - DO NOT say 'unrelated'.\n"
        "   - Instead use soft, academic phrasing such as:\n"
        "       * 'conceptually distinct within the course progression'\n"
        "       * 'address different layers of the broader networking domain'\n"
        "       * 'connected only at a high level through the overarching subject'\n"
        "       * 'serve different educational purposes within the curriculum'\n"
        "5. Base ALL reasoning strictly on PREVIEW TEXT.\n"
        "6. Never invent details that are not supported by the preview.\n"
        "7. Never mention embeddings, similarity scores, model behaviour, or generic wording.\n\n"

        "STYLE REQUIREMENTS FOR EACH RELATIONSHIP:\n"
        "- If topics overlap:\n"
        "    * Write AT LEAST 6 sentences and up to 10 sentences.\n"
        "    * Use precise academic language (as in lecture notes, textbooks, or research explanations).\n"
        "    * Mention SPECIFIC concepts, terms, models, or mechanisms present in BOTH documents.\n"
        "    * Explain how one document provides theoretical foundations and how the other applies or extends them.\n"
        "    * Describe the conceptual progression (e.g., introductory → intermediate → advanced).\n"
        "- If topics are distinct:\n"
        "    * Write 2–4 sentences using soft academic phrasing (NO 'unrelated').\n"
        "    * Explain how their purposes, focus areas, or levels diverge within the broader subject.\n\n"

        "DOCUMENTS (with previews):\n"
        "===========================\n"
        f"{chr(10).join(doc_blocks)}\n\n"

        "PAIRS TO ANALYZE:\n"
        "==================\n"
        f"{chr(10).join(pair_names)}\n\n"

        "Return STRICT JSON ONLY in this format:\n"
        "{\n"
        '  "topics": {\n'
        '      "docname": "primary topic (1–3 words)",\n'
        "      ...\n"
        "  },\n"
        '  "global_themes": ["theme1", "theme2", ...],\n'
        '  "relations": [\n'
        "    {\n"
        '      "doc_a": "exact document name",\n'
        '      "doc_b": "exact document name",\n'
        '      "relationship": "A detailed explanation following the rules above."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Output ONLY valid JSON.\n"
    )


    llm = LLMClient()
    raw = llm.complete(prompt, model=model)
    data = _safe_json_object(raw)

    # 3) Coerce into models
    global_themes = data.get("global_themes") or []
    if not isinstance(global_themes, list):
        global_themes = [str(global_themes)]

    relations_raw = data.get("relations") or []

    # build lookup of our numeric similarities so we can attach them
    sim_lookup: Dict[tuple[str, str], float] = {}
    for p in filtered_pairs:
        a = p["doc_a"]
        b = p["doc_b"]
        sim = float(p["similarity"])
        sim_lookup[(a, b)] = sim
        sim_lookup[(b, a)] = sim  # symmetric

    relation_models: List[DocPairRelation] = []
    if isinstance(relations_raw, list):
        for item in relations_raw:
            if not isinstance(item, dict):
                continue
            a = str(item.get("doc_a") or "").strip()
            b = str(item.get("doc_b") or "").strip()
            if not a or not b:
                continue
            rel_text = str(item.get("relationship") or "").strip()

            sim = sim_lookup.get((a, b), 0.0)

            relation_models.append(
                DocPairRelation(
                    doc_a=a,
                    doc_b=b,
                    similarity=sim,
                    relationship=rel_text,
                )
            )

    return CrossDocRelations(
        documents=docs,
        global_themes=[str(t) for t in global_themes],
        relations=relation_models,
    )
