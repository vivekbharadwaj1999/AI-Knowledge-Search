import json
import re
from typing import Any, Dict, List, Optional
from app.config import LLMClient

def _build_insights_prompt(question: str, answer: str, context: List[str]) -> str:
    joined_context = "\n\n---\n\n".join(context)
    return f"""Analyze this question-answer pair and its context.

Question: {question}

Answer: {answer}

Context: {joined_context}

You must respond with ONLY a valid JSON object. Do not include any text before or after the JSON.

Required JSON structure:
{{
  "summary": "comprehensive summary of the answer in 4-6 sentences",
  "key_points": ["point 1", "point 2", "point 3"],
  "entities": ["entity1", "entity2"],
  "suggested_questions": ["question 1?", "question 2?", "question 3?"],
  "mindmap": "Topic\\n- Point 1\\n- Point 2",
  "reading_difficulty": "beginner",
  "sentiment": "neutral",
  "keywords": ["keyword1", "keyword2"],
  "sentence_importance": [
    {{"sentence": "exact sentence from context", "score": 5}}
  ]
}}

CRITICAL RULES:

For summary:
- Write EXACTLY 4-6 complete sentences (not more, not less)
- Each sentence must end with proper punctuation (. ! ?)
- Cover the main points comprehensively
- Include key facts, numbers, and specifics from the answer
- Make it informative and detailed enough to understand the answer without reading it
- DO NOT truncate mid-sentence

For entities:
- Extract ALL named entities from the answer and context
- Include: people, organizations, companies, universities, institutions
- Include: technologies, frameworks, programming languages, tools, platforms
- Include: specific projects, products, systems mentioned by name
- Include: locations, cities, countries
- Include: degrees, certifications, qualifications
- Include: time periods, dates, durations (e.g., "3.5 years")
- Provide 10-12 entities maximum (6 to 8 is ideal)

For sentence_importance:
- score 5: sentence content was directly used or paraphrased in the answer
- score 4: sentence provided specific facts/data that appear in the answer
- score 3: sentence provided essential background for the answer
- score 2: sentence was tangentially related
- score 1: sentence was minimally relevant
- score 0: sentence was not used

Include only sentences with score >= 3. Maximum 15 sentences.
Copy sentences EXACTLY as they appear in context.

For keywords: include meaningful technical terms, domain-specific words, important concepts, and key nouns from the context. Focus on words that someone searching for this information would use. Include 10-20 keywords.

Respond with ONLY the JSON object, nothing else."""

def _extract_json_from_response(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    
    if raw.startswith("```"):
        lines = raw.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                code_lines.append(line)
        raw = "\n".join(code_lines)
    
    raw = re.sub(r'^[^{]*', '', raw)
    raw = re.sub(r'[^}]*$', '', raw)
    
    start = raw.find("{")
    end = raw.rfind("}")
    
    if start == -1 or end == -1 or end <= start:
        return {}
    
    snippet = raw[start: end + 1]
    
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        try:
            snippet = re.sub(r',(\s*[}\]])', r'\1', snippet)
            return json.loads(snippet)
        except json.JSONDecodeError:
            return {}

def _get_default_insights(question: str, answer: str) -> Dict[str, Any]:
    words = answer.split()
    summary = " ".join(words[:50]) + ("..." if len(words) > 50 else "")
    
    return {
        "summary": summary,
        "key_points": ["Information extracted from context"],
        "entities": [],
        "suggested_questions": ["Can you provide more details?"],
        "mindmap": "Main Topic\n- Key Point",
        "reading_difficulty": "intermediate",
        "sentiment": "neutral",
        "keywords": list(set([w.lower() for w in question.split() if len(w) > 3][:10])),
        "sentence_importance": []
    }

def generate_insights(
    question: str,
    answer: str,
    context: List[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    llm = LLMClient()
    prompt = _build_insights_prompt(question, answer, context)
    
    try:
        raw = llm.complete(prompt, model=model)
        data = _extract_json_from_response(raw)
        
        if not data or not isinstance(data, dict):
            return _get_default_insights(question, answer)
        
    except Exception:
        return _get_default_insights(question, answer)

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

    summary = str(data.get("summary", "") or "")
    if not summary:
        summary = _get_default_insights(question, answer)["summary"]

    key_points = as_list_of_str(data.get("key_points", []))
    if not key_points:
        key_points = ["Information from context"]
        
    entities = as_list_of_str(data.get("entities", []))
    suggested_questions = as_list_of_str(data.get("suggested_questions", []))
    if not suggested_questions:
        suggested_questions = ["Can you elaborate on this?"]
        
    keywords = as_list_of_str(data.get("keywords", []))
    if not keywords:
        keywords = list(set([w.lower() for w in question.split() if len(w) > 3][:10]))

    mindmap_raw = data.get("mindmap", "")
    if isinstance(mindmap_raw, list):
        flat = as_list_of_str(mindmap_raw)
        mindmap = "\n".join(flat)
    else:
        mindmap = str(mindmap_raw or "")
    
    if not mindmap:
        mindmap = "Topic\n- Key Information"

    reading_difficulty = str(data.get("reading_difficulty", "") or "intermediate")
    sentiment = str(data.get("sentiment", "") or "neutral")

    highlights_raw = data.get("highlights", [])
    highlights: List[List[str]] = []
    if isinstance(highlights_raw, list):
        for item in highlights_raw:
            highlights.append(as_list_of_str(item))
    else:
        highlights = []

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
            if score >= 3:
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