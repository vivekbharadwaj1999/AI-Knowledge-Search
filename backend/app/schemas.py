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


class ReportSection(BaseModel):
    heading: str
    content: str


class QAItem(BaseModel):
    question: str
    answer: str


class KnowledgeGraphEdge(BaseModel):
    source: str
    relation: str
    target: str


class DocumentReport(BaseModel):
    # we use doc_name (same as elsewhere in your app)
    doc_name: str
    title: Optional[str] = None

    executive_summary: str
    sections: List[ReportSection]

    key_concepts: List[str]
    concept_explanations: List[str]

    relationships: List[str]
    knowledge_graph: List[KnowledgeGraphEdge]

    practice_questions: List[QAItem]

    difficulty_level: str
    difficulty_explanation: str

    study_path: List[str]

    explain_like_im_5: str
    cheat_sheet: List[str]


class ReportRequest(BaseModel):
    doc_name: str
    # optional override for model (same pattern as AskRequest)
    model: Optional[str] = None


class DocPairRelation(BaseModel):
    doc_a: str
    doc_b: str
    similarity: float
    relationship: str


class CrossDocRelations(BaseModel):
    documents: List[str]
    global_themes: List[str]
    relations: List[DocPairRelation]


class CrossDocRelationsRequest(BaseModel):
    # optional model override, same idea as AskRequest
    model: Optional[str] = None

# Prompt coaching / critique


class CritiqueScores(BaseModel):
    correctness: Optional[float] = None
    completeness: Optional[float] = None
    clarity: Optional[float] = None
    hallucination_risk: Optional[float] = None


class CritiqueRequest(BaseModel):
    question: str
    answer_model: str
    critic_model: Optional[str] = None
    top_k: int = 5
    doc_name: Optional[str] = None


class CritiqueResponse(BaseModel):
    question: str
    answer_model: str
    critic_model: str

    answer: str
    context: List[str]
    sources: List[SourceChunk]

    answer_critique_markdown: str
    prompt_feedback_markdown: str
    improved_prompt: str
    prompt_issue_tags: List[str] = []

    scores: Optional[CritiqueScores] = None
