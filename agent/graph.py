# agent/graph.py
from langgraph.graph import StateGraph, END

from agent.state import GraphState
from rag.embeddings import get_embeddings_model
from rag.vectorstore import build_faiss_index, query_index
from rag.generator import generate_answer
from agent.validator import validate_with_llm
from langgraph.graph import StateGraph, END
from agent.state import GraphState
from agent.retrieval import retrieve_kb, retrieve_tickets, ticket_docs_to_cases
from rag.generator import generate_answer
from agent.validator import validate_with_llm

# --- For now we reuse the tiny in-memory docs to learn the loop.
# In the next phase, this becomes a real ingestion pipeline + persistent FAISS.
from langchain_core.documents import Document

TOY_DOCS = [
    Document(page_content="Refunds take 5-10 business days. EU refunds may take longer.", metadata={"source":"policy.md", "chunk_id": 0}),
    Document(page_content="If a user is locked out, advise password reset and verify email.", metadata={"source":"runbook.md", "chunk_id": 0}),
    Document(page_content="Escalate billing issues if charge is duplicated or pending > 7 days.", metadata={"source":"runbook.md", "chunk_id": 1}),
]

_embeddings = get_embeddings_model()
_index = build_faiss_index(TOY_DOCS, _embeddings)

def strip_unsupported_escalation(answer_obj, kb_docs):
    kb_text = "\n".join(d.page_content for d in kb_docs).lower()
    if "escalat" in answer_obj.answer.lower() and "escalat" not in kb_text:
        # Remove escalation language and downgrade confidence
        answer_obj.answer = answer_obj.answer.replace("escalate", "follow up with support")
        answer_obj.confidence = "low" if answer_obj.confidence == "high" else answer_obj.confidence
        if not answer_obj.missing_info:
            answer_obj.missing_info = "Escalation steps are not provided in the knowledge base for this issue."
    return answer_obj

def retrieve_node(state: GraphState) -> GraphState:
    q = state["query"]
    state["kb_evidence"] = retrieve_kb(q, k=state["kb_k"])
    state["ticket_evidence"] = retrieve_tickets(q, k=state["tickets_k"])
    return state

def generate_node(state: GraphState) -> GraphState:
    cases = ticket_docs_to_cases(state["ticket_evidence"])
    state["answer"] = generate_answer(state["question"], state["kb_evidence"], cases)
    state["answer"] = strip_unsupported_escalation(state["answer"], state["kb_evidence"])
    return state

# validate_node can stay the same, but we should pass kb evidence into the validator
def validate_node(state: GraphState) -> GraphState:
    answer_json = state["answer"].model_dump_json()
    result = validate_with_llm(state["question"], answer_json, state["kb_evidence"])

    state["decision"] = result.decision
    state["validator_feedback"] = result.feedback

    if result.decision == "RETRY_WITH_MORE_CONTEXT":
        state["query"] = result.suggested_query or state["query"]
        state["kb_k"] = result.suggested_k or max(state["kb_k"], 8)
        state["retries"] = state["retries"] + 1

    return state

def route_after_validate(state: GraphState):
    if state["decision"] == "PASS":
        return END
    if state["decision"] == "REFUSE":
        return END
    # RETRY
    if state["retries"] >= 1:
        # one retry max for today (bounded agent, production-style)
        return END
    return "retrieve"

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("validate", validate_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", route_after_validate)

    return g.compile()
