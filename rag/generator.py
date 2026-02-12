# rag/generator.py
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from rag.schema import RagAnswer

def get_llm(model: str = "gpt-4o-mini"):
    # We'll later swap in a fine-tuned model name here
    return ChatOpenAI(model=model, temperature=0)

def format_evidence(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        cid = d.metadata.get("chunk_id", -1)
        lines.append(f"[{src}#{cid}] {d.page_content}")
    return "\n".join(lines)

from rag.schema import RagAnswer

def format_kb_evidence(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        cid = d.metadata.get("chunk_id", -1)
        lines.append(f"CITE_KEY=({src},{cid})\n{d.page_content}")
    return "\n\n".join(lines)

def format_cases(cases: list[dict]) -> str:
    lines = []
    for c in cases:
        lines.append(f"- (row_id={c['row_id']}) Customer: {c['customer']} | Support: {c['support']}")
    return "\n".join(lines)

def generate_answer(question: str, kb_docs, ticket_cases) -> RagAnswer:
    kb_text = format_kb_evidence(kb_docs)
    cases_text = format_cases(ticket_cases)

    llm = get_llm().with_structured_output(RagAnswer)

    system = SystemMessage(content=(
        "You are a support copilot.\n"
        "Rules:\n"
        "1) Use ONLY KB_EVIDENCE to make factual/policy claims.\n"
        "2) citations must refer ONLY to KB_EVIDENCE chunks.\n"
        "3) You may use SIMILAR_CASES only as examples of how support responded before, not as authority.\n"
        "4) If KB_EVIDENCE is insufficient, set confidence='low' and explain missing_info.\n"
        "5) The citation source must be EXACTLY one of the source values inside the CITE_KEY tuples. Never use 'KB_EVIDENCE' or 'Source'.\n"
    ))

    user = HumanMessage(content=(
        f"QUESTION:\n{question}\n\n"
        f"KB_EVIDENCE:\n{kb_text}\n\n"
        f"SIMILAR_CASES (examples, not authority):\n{cases_text}\n"
    ))

    return llm.invoke([system, user])
