# agent/retrieval.py
from __future__ import annotations
from typing import List
from rag.embeddings import get_embeddings_model
from rag.vectorstore import load_faiss_index

_embeddings = get_embeddings_model()
_kb = load_faiss_index("data/index/kb_faiss", _embeddings)
_tickets = load_faiss_index("data/index/tickets_faiss", _embeddings)

def retrieve_kb(query: str, k: int = 5):
    return _kb.similarity_search(query, k=k)

def retrieve_tickets(query: str, k: int = 3):
    return _tickets.similarity_search(query, k=k)

def ticket_docs_to_cases(ticket_docs) -> List[dict]:
    cases = []
    for d in ticket_docs:
        rid = int(d.metadata.get("row_id", -1))
        text = d.page_content.splitlines()
        cust = text[0].replace("Customer: ", "") if text else ""
        supp = text[1].replace("Support: ", "") if len(text) > 1 else ""
        cases.append({"row_id": rid, "customer": cust, "support": supp})
    return cases
