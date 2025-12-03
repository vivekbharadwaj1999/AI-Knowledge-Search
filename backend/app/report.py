import json
from typing import Any, Dict, List, Optional

from app.config import LLMClient
from app.vector_store import get_document_text
from app.schemas import (
    DocumentReport,
    ReportSection,
    KnowledgeGraphEdge,
    QAItem,
)

MAX_REPORT_SOURCE_CHARS = 20000

CHUNK_SIZE_CHARS = 4000

SAFE_MAX_PROMPT_CHARS = 6000


def _chunk_text_by_chars(text: str, max_chars: int) -> List[str]:
    """Split text into chunks of at most max_chars, trying to cut at paragraph boundaries."""
    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        cut = text.rfind("\n\n", start, end)
        if cut == -1:
            cut = text.rfind(". ", start, end)
        if cut == -1 or cut <= start:
            cut = end

        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = cut

    return chunks


def _summarize_long_document(
    full_text: str,
    llm: LLMClient,
    *,
    model: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE_CHARS,
) -> str:
    """
    Use the full document but process it in multiple LLM calls:
    - Summarize each chunk separately.
    - Concatenate the partial summaries into a 'meta-document'.
    """
    parts = _chunk_text_by_chars(full_text, chunk_size)
    if len(parts) == 1:
        return parts[0]

    summaries: List[str] = []

    for idx, part in enumerate(parts, start=1):
        prompt = f"""
You are summarizing part {idx} of {len(parts)} of a longer technical document.

PART {idx} CONTENT:
-------------------
{part}
-------------------

Write a concise summary of this part focusing on:
- main ideas
- important definitions or equations
- key arguments or results

Use 2–5 short paragraphs. Do NOT add JSON, backticks, or metadata.
"""
        summary = llm.complete(prompt, model=model)
        summaries.append(f"PART {idx} SUMMARY:\n{summary.strip()}")

    combined = "\n\n".join(summaries)
    return combined


def _truncate_for_prompt(text: str, max_chars: int = SAFE_MAX_PROMPT_CHARS) -> str:
    """
    Final safety truncation before sending text to the JSON-report prompt.
    This helps avoid hitting the provider's token limits even for large docs.
    """
    if len(text) <= max_chars:
        return text

    cut = text.rfind(".", 0, max_chars)
    if cut == -1:
        cut = max_chars

    return (
        text[:cut].strip()
        + "\n\n[Note: Source text was truncated slightly to fit within model limits.]"
    )


def _safe_parse_json_object(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    snippet = raw[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return {}


def _build_report_prompt(doc_text: str) -> str:
    return f"""
You are an expert tutor and study coach.

You are given the content of a single document (such as lecture notes, a research paper,
slides, or a technical article). Your job is to turn it into a detailed interactive study report.

DOCUMENT CONTENT:
-----------------
{doc_text}
-----------------

Return a STRICT JSON object with the following keys:

- executive_summary: 1–2 paragraphs high-level summary of the document.
- sections: array of objects, each with:
    - heading: short heading (invent if necessary).
    - content: 1–3 paragraphs explaining that section of the document.
- key_concepts: array of short concept names.
- concept_explanations: array of same length as key_concepts, each a 2–3 sentence explanation.
- relationships: array of 1–3 sentence strings describing how 2–4 concepts relate.
- knowledge_graph: array of objects, each with:
    - source: concept name (string),
    - relation: short verb phrase (string),
    - target: concept name (string).
- practice_questions: array of objects, each with:
    - question: short knowledge-check question (string),
    - answer: concise answer (string).
- difficulty_level: one of "beginner", "intermediate", "advanced".
- difficulty_explanation: 2–4 sentences explaining why that difficulty was chosen.
- study_path: ordered array of bullet-point strings for how to study this document.
- explain_like_im_5: simplified explanation in one paragraph as if to a 5-year-old.
- cheat_sheet: array of bullet-point strings summarising key formulas, facts, or steps.

CRITICAL RULES:
- Output ONLY a single valid JSON object, with no backticks or extra commentary.
- If you are unsure, make reasonable assumptions but stay consistent.
"""


def generate_document_report(
    doc_name: str,
    *,
    model: Optional[str] = None,
    max_chars: int = MAX_REPORT_SOURCE_CHARS,
) -> DocumentReport:
    """
    Generate a rich study report for a document.

    - For smaller documents (len <= max_chars), use the raw text directly.
    - For larger ones, first summarize in chunks, then build the report from the combined summary.
    """
    full_text = get_document_text(doc_name, max_chars=None)

    llm = LLMClient()

    if len(full_text) <= max_chars:
        source_text = full_text
    else:
        source_text = _summarize_long_document(
            full_text,
            llm,
            model=model,
            chunk_size=CHUNK_SIZE_CHARS,
        )

    source_text = _truncate_for_prompt(source_text)

    prompt = _build_report_prompt(source_text)
    raw = llm.complete(prompt, model=model)

    data = _safe_parse_json_object(raw)

    def as_list_of_str(value: Any) -> List[str]:
        out: List[str] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (str, int, float)):
                    out.append(str(item))
                elif isinstance(item, dict):
                    parts = []
                    for k, v in item.items():
                        parts.append(f"{k}: {v}")
                    if parts:
                        out.append("; ".join(parts))
                else:
                    out.append(str(item))
        return out

    executive_summary = str(data.get("executive_summary", "") or "")

    sections_raw = data.get("sections") or []
    sections: List[ReportSection] = []
    if isinstance(sections_raw, list):
        for sec in sections_raw:
            if not isinstance(sec, dict):
                continue
            heading = str(sec.get("heading", "") or "")
            content = str(sec.get("content", "") or "")
            if heading or content:
                sections.append(ReportSection(heading=heading, content=content))

    key_concepts = as_list_of_str(data.get("key_concepts") or [])
    concept_explanations = as_list_of_str(data.get("concept_explanations") or [])
    relationships = as_list_of_str(data.get("relationships") or [])

    kg_raw = data.get("knowledge_graph") or []
    knowledge_graph: List[KnowledgeGraphEdge] = []
    if isinstance(kg_raw, list):
        for edge in kg_raw:
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("source", "") or "")
            rel = str(edge.get("relation", "") or "")
            tgt = str(edge.get("target", "") or "")
            if src and rel and tgt:
                knowledge_graph.append(
                    KnowledgeGraphEdge(source=src, relation=rel, target=tgt)
                )

    qa_raw = data.get("practice_questions") or []
    practice_questions: List[QAItem] = []
    if isinstance(qa_raw, list):
        for qa in qa_raw:
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question", "") or "")
            answer = str(qa.get("answer", "") or "")
            if question or answer:
                practice_questions.append(QAItem(question=question, answer=answer))

    difficulty_level = str(data.get("difficulty_level", "") or "")
    difficulty_explanation = str(data.get("difficulty_explanation", "") or "")

    study_path = as_list_of_str(data.get("study_path") or [])
    explain_like_im_5 = str(data.get("explain_like_im_5", "") or "")
    cheat_sheet = as_list_of_str(data.get("cheat_sheet") or [])

    return DocumentReport(
        doc_name=doc_name,
        executive_summary=executive_summary,
        sections=sections,
        key_concepts=key_concepts,
        concept_explanations=concept_explanations,
        relationships=relationships,
        knowledge_graph=knowledge_graph,
        practice_questions=practice_questions,
        difficulty_level=difficulty_level,
        difficulty_explanation=difficulty_explanation,
        study_path=study_path,
        explain_like_im_5=explain_like_im_5,
        cheat_sheet=cheat_sheet,
    )
