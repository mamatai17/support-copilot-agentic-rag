# agent/validator.py
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
from pydantic import BaseModel
from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent.hard_checks import (
    hard_check_citations_whitelist,
    hard_check_domain_mismatch,
    hard_check_action_grounding,
)

Decision = Literal["PASS", "RETRY_WITH_MORE_CONTEXT", "REFUSE"]


class ValidationResult(BaseModel):
    decision: Decision
    feedback: str
    suggested_query: Optional[str] = None
    suggested_k: Optional[int] = None


def format_evidence(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        cid = d.metadata.get("chunk_id", -1)
        lines.append(f"[{src}#{cid}] {d.page_content}")
    return "\n".join(lines)


def validate_with_llm(question: str, answer_json: str, kb_evidence_docs) -> ValidationResult:
    """
    Two-stage validation:
      1) Deterministic hard checks (citation whitelist + domain mismatch + basic action grounding)
      2) LLM-as-judge for nuanced faithfulness
    """
    import json

    try:
        ans_obj = json.loads(answer_json)
        answer_text = ans_obj.get("answer", "")
    except Exception:
        return ValidationResult(
            decision="RETRY_WITH_MORE_CONTEXT",
            feedback="Answer JSON could not be parsed; retrying.",
            suggested_query=question,
            suggested_k=10,
        )

    hard_errors = []
    hard_errors += hard_check_citations_whitelist(ans_obj, kb_evidence_docs)
    hard_errors += hard_check_domain_mismatch(question, kb_evidence_docs)
    hard_errors += hard_check_action_grounding(question, answer_text, kb_evidence_docs)

    if os.getenv("DEBUG_VALIDATOR", "0") == "1":
        print("\n[VALIDATOR DEBUG]")
        print("Question:", question)
        print("KB evidence count:", len(kb_evidence_docs))
        if kb_evidence_docs:
            print("KB[0] metadata:", kb_evidence_docs[0].metadata)
            print("KB[0] text head:", kb_evidence_docs[0].page_content[:120].replace("\n", " "), "...")
        print("Hard errors:", hard_errors)

    if hard_errors:
        return ValidationResult(
            decision="RETRY_WITH_MORE_CONTEXT",
            feedback="Hard check failed: " + " | ".join(hard_errors),
            suggested_query=question,
            suggested_k=10,
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(ValidationResult)

    system = SystemMessage(content=(
        "You are a strict QA validator for a RAG system.\n"
        "PASS only if EVERY actionable instruction is directly supported by KB evidence.\n"
        "If the answer introduces any policy/threshold not explicitly stated, choose RETRY_WITH_MORE_CONTEXT.\n"
        "If the KB evidence is about the wrong domain/topic relative to the question, choose RETRY_WITH_MORE_CONTEXT.\n"
        "When RETRY, propose suggested_k >= 8."
    ))

    user = HumanMessage(content=(
        f"QUESTION:\n{question}\n\n"
        f"ANSWER_JSON:\n{answer_json}\n\n"
        f"KB_EVIDENCE:\n{format_evidence(kb_evidence_docs)}\n\n"
        "Is the answer fully supported and on-topic?"
    ))

    return llm.invoke([system, user])
