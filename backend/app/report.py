# backend/app/report.py
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


def _safe_parse_json_object(raw: str) -> Dict[str, Any]:
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
    max_chars: int = 20000,
) -> DocumentReport:
    raw_text = get_document_text(doc_name, max_chars=max_chars)

    llm = LLMClient()
    prompt = _build_report_prompt(raw_text)
    raw = llm.complete(prompt, model=model)

    data = _safe_parse_json_object(raw)

    def as_list_of_str(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        out: List[str] = []
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

    sections_raw = data.get("sections", [])
    sections: List[ReportSection] = []
    if isinstance(sections_raw, list):
        for item in sections_raw:
            if isinstance(item, dict):
                heading = str(item.get("heading", "")
                              or "").strip() or "Section"
                content_val = item.get("content", "") or ""
                if isinstance(content_val, list):
                    content = "\n\n".join(str(x) for x in content_val)
                else:
                    content = str(content_val)
                content = content.strip()
            else:
                heading = "Section"
                content = str(item)
            sections.append(ReportSection(heading=heading, content=content))

    key_concepts = as_list_of_str(data.get("key_concepts", []))
    concept_explanations = as_list_of_str(data.get("concept_explanations", []))

    relationships = as_list_of_str(data.get("relationships", []))

    kg_raw = data.get("knowledge_graph", [])
    knowledge_graph: List[KnowledgeGraphEdge] = []
    if isinstance(kg_raw, list):
        for edge in kg_raw:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", "") or "").strip()
            relation = str(edge.get("relation", "") or "").strip()
            target = str(edge.get("target", "") or "").strip()
            if not source or not target:
                continue
            knowledge_graph.append(
                KnowledgeGraphEdge(
                    source=source, relation=relation, target=target)
            )

    pq_raw = data.get("practice_questions", [])
    practice_questions: List[QAItem] = []
    if isinstance(pq_raw, list):
        for qa in pq_raw:
            if not isinstance(qa, dict):
                continue
            q = str(qa.get("question", "") or "").strip()
            a = str(qa.get("answer", "") or "").strip()
            if not q:
                continue
            practice_questions.append(QAItem(question=q, answer=a))

    difficulty_level = (
        str(data.get("difficulty_level", "") or "").strip() or "intermediate"
    )
    difficulty_explanation = str(
        data.get("difficulty_explanation", "") or ""
    ).strip()

    study_path = as_list_of_str(data.get("study_path", []))

    explain_like_im_5 = str(data.get("explain_like_im_5", "") or "").strip()
    cheat_sheet = as_list_of_str(data.get("cheat_sheet", []))

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
