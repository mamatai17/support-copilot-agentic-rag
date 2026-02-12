from typing import TypedDict, List, Optional, Literal
from langchain_core.documents import Document
from rag.schema import RagAnswer

Decision = Literal["PASS", "RETRY_WITH_MORE_CONTEXT", "REFUSE"]

class GraphState(TypedDict):
    question: str
    query: str
    kb_k: int
    tickets_k: int
    kb_evidence: List[Document]
    ticket_evidence: List[Document]
    answer: Optional[RagAnswer]
    decision: Optional[Decision]
    validator_feedback: Optional[str]
    retries: int
