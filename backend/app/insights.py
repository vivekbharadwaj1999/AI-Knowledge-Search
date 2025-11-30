# backend/app/insights.py
import json
from typing import Any, Dict, List, Optional

from app.config import LLMClient


def _build_insights_prompt(question: str, answer: str, context: List[str]) -> str:
    joined_context = "\n\n---\n\n".join(context)
    return f"""
You are an assistant that analyzes an answer and its supporting context.

Question:
{question}

Answer:
{answer}

Context:
{joined_context}

Return a STRICT JSON object with the following keys:

- summary: short summary of the answer (string)
- key_points: list of 3-7 key bullet points (array of strings)
- entities: list of important entities (array of strings)
- suggested_questions: list of 3-5 follow-up questions (array of strings)
- mindmap: compact text representation of a mindmap (string)
- reading_difficulty: one of "beginner", "intermediate", "advanced" (string)
- sentiment: short sentiment label like "neutral", "positive", "critical" (string)
- keywords: list of important keywords/phrases (array of strings)
- highlights: OPTIONAL list of groups of phrases to highlight (array of array of strings)
- sentence_importance: array of objects, each with:
    - sentence: an important sentence from the CONTEXT (string)
    - score: integer from 0 to 5 indicating importance
      (5 = absolutely key, 3 = helpful, 1 = minor, 0 = irrelevant)

IMPORTANT:
- Only return valid JSON, no commentary.
- For sentence_importance, include at most 15 sentences, pick those that
  best explain or support the answer for this question.
"""


def _safe_parse_json_object(raw: str) -> Dict[str, Any]:
    """
    Try to extract a JSON object from the LLM output.
    If parsing fails, return {} so we can fall back to defaults.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    snippet = raw[start: end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return {}


def generate_insights(
    question: str,
    answer: str,
    context: List[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    llm = LLMClient()
    prompt = _build_insights_prompt(question, answer, context)
    raw = llm.complete(prompt, model=model)

    data = _safe_parse_json_object(raw)

    # Helper: turn any list-like into list[str]
    def as_list_of_str(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, (int, float)):
                out.append(str(item))
            elif isinstance(item, dict):
                # LLM sometimes returns {"name": "...", "type": "..."}
                name = item.get("name")
                t = item.get("type") or item.get("category")
                if name and t:
                    out.append(f"{name} ({t})")
                elif name:
                    out.append(str(name))
                else:
                    out.append(str(item))
            else:
                out.append(str(item))
        return out

    # --- main scalar fields ---
    summary = str(data.get("summary", "") or "")

    key_points = as_list_of_str(data.get("key_points", []))
    entities = as_list_of_str(data.get("entities", []))
    suggested_questions = as_list_of_str(data.get("suggested_questions", []))
    keywords = as_list_of_str(data.get("keywords", []))

    # mindmap can be string OR list/structure → turn into a single string
    mindmap_raw = data.get("mindmap", "")
    if isinstance(mindmap_raw, list):
        # flatten any nested list/objects to strings
        flat = as_list_of_str(mindmap_raw)
        mindmap = "\n".join(flat)
    else:
        mindmap = str(mindmap_raw or "")

    reading_difficulty = str(data.get("reading_difficulty", "") or "")
    sentiment = str(data.get("sentiment", "") or "")

    # --- optional LLM-driven highlights: List[List[str]] ---
    highlights_raw = data.get("highlights", [])
    highlights: List[List[str]] = []
    if isinstance(highlights_raw, list):
        for item in highlights_raw:
            highlights.append(as_list_of_str(item))
    else:
        highlights = []

    # sentence_importance: [{ sentence, score }]
    sent_raw = data.get("sentence_importance", [])
    sentence_importance: List[Dict[str, Any]] = []
    if isinstance(sent_raw, list):
        for item in sent_raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("sentence", "") or "").strip()
            if not text:
                continue
            try:
                score = int(item.get("score", 0))
            except (TypeError, ValueError):
                score = 0
            score = max(0, min(5, score))
            sentence_importance.append({"sentence": text, "score": score})

    return {
        "summary": summary,
        "key_points": key_points,
        "entities": entities,
        "suggested_questions": suggested_questions,
        "mindmap": mindmap,
        "reading_difficulty": reading_difficulty,
        "sentiment": sentiment,
        "keywords": keywords,
        "highlights": highlights,
        "sentence_importance": sentence_importance,
    }
