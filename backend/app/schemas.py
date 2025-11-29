from typing import List, Optional
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_name: Optional[str] = None  # 🔑 new


class AskResponse(BaseModel):
    answer: str
    context: List[str]
