from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_name: Optional[str] = None
    model: Optional[str] = None
    similarity: Optional[str] = None
    normalize_vectors: bool = True
    embedding_model: Optional[str] = None


class SourceChunk(BaseModel):
    doc_name: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    context: List[str]
    model_used: str
    sources: List[SourceChunk] = []


class InsightsRequest(BaseModel):
    question: str
    answer: str
    context: List[str]
    model: Optional[str] = None


class SentenceImportance(BaseModel):
    sentence: str
    score: int


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
    model: Optional[str] = None
    similarity: Optional[str] = "cosine"
    normalize_vectors: bool = True
    max_pairs: int = 12
    min_similarity: float = 0.2


class CritiqueScores(BaseModel):
    correctness: Optional[float] = None
    completeness: Optional[float] = None
    clarity: Optional[float] = None
    hallucination_risk: Optional[float] = None
    prompt_quality: Optional[float] = None


class CritiqueRound(BaseModel):
    round: int
    question: str
    answer: str
    context: List[str]
    sources: Optional[List[Dict[str, Any]]] = None
    answer_critique_markdown: str
    prompt_feedback_markdown: str
    improved_prompt: str
    prompt_issue_tags: List[str] = []
    scores: Optional[CritiqueScores] = None


class CritiqueRequest(BaseModel):
    question: str
    answer_model: str
    critic_model: Optional[str] = None
    top_k: int = 5
    doc_name: Optional[str] = None
    self_correct: bool = False
    similarity: Optional[str] = "cosine"
    normalize_vectors: bool = True
    embedding_model: Optional[str] = None


class CritiqueResponse(BaseModel):
    question: str
    answer_model: str
    critic_model: str
    answer: str
    context: List[str]
    sources: Optional[List[Dict[str, Any]]] = None
    answer_critique_markdown: str
    prompt_feedback_markdown: str
    improved_prompt: str
    prompt_issue_tags: List[str] = []
    scores: Optional[CritiqueScores] = None
    rounds: List[CritiqueRound] = []


class CritiqueLogRow(BaseModel):
    timestamp: Optional[str] = None
    question: Optional[str] = None
    answer_model: str
    critic_model: str
    doc_name: Optional[str] = None
    self_correct: bool
    similarity: Optional[str] = None
    num_rounds: int

    r1_correctness: Optional[float] = None
    r1_hallucination: Optional[float] = None
    rN_correctness: Optional[float] = None
    rN_hallucination: Optional[float] = None
    delta_correctness: Optional[float] = None
    delta_hallucination: Optional[float] = None


class CritiqueLogResponse(BaseModel):
    rows: List[CritiqueLogRow]


# Authentication schemas
class SignupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str
    is_guest: bool


class UserResponse(BaseModel):
    username: str
    is_guest: bool
