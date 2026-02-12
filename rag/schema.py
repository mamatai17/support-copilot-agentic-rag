# rag/schema.py
from pydantic import BaseModel
from typing import List, Optional

class Citation(BaseModel):
    source: str
    chunk_id: int

class SimilarCase(BaseModel):
    row_id: int
    customer: str
    support: str

class RagAnswer(BaseModel):
    answer: str
    citations: List[Citation]          # authoritative citations (KB only)
    confidence: str                    # "high" | "medium" | "low"
    missing_info: Optional[str] = None
    next_steps: List[str]
    similar_cases: List[SimilarCase]   # examples from ticket memory
