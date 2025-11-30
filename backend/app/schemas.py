from typing import List, Optional
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_name: Optional[str] = None
    # 🔧 optional override for the LLM model (e.g. "llama-3.1-8b-instant")
    model: Optional[str] = None


class SourceChunk(BaseModel):
    doc_name: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    context: List[str]
    # which model actually answered (defaults to GROQ_MODEL)
    model_used: str
    # detailed per-chunk metadata (doc + score)
    sources: List[SourceChunk] = []


class InsightsRequest(BaseModel):
    # we generate insights *about* a specific QA turn
    question: str
    answer: str
    context: List[str]
    model: Optional[str] = None  # allow overriding model for insights too


class SentenceImportance(BaseModel):
    sentence: str
    score: int  # 0–5


class InsightsResponse(BaseModel):
    summary: str
    key_points: List[str]
    entities: List[str]
    suggested_questions: List[str]
    mindmap: str
    reading_difficulty: str
    sentiment: str
    keywords: List[str]
    highlights: List[List[str]] = []
    sentence_importance: List[SentenceImportance] = []
