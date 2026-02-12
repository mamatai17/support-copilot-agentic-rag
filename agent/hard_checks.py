# agent/hard_checks.py
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any

# Simple topic hints (extend later)
TOPIC_KEYWORDS = {
    "tax_refund": [
        "irs", "irs2go", "where's my refund", "wheres my refund",
        "1040", "w-2", "tax year", "efile", "e-file", "tax return",
        "state tax", "tax commission"
    ],
    "commerce_refund": ["order", "purchase", "merchant", "store", "card", "bank", "amazon", "refund", "return"],
    "network": ["lte", "signal", "bars", "data", "internet", "load", "speed"],
}

def _contains_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)

def infer_question_domain(question: str) -> str:
    q = question.lower()
    if _contains_any(q, TOPIC_KEYWORDS["tax_refund"]):
        return "tax_refund"
    # if question says refund but not tax-y, treat as commerce by default
    if "refund" in q or "refunds" in q:
        return "commerce_refund"
    if _contains_any(q, TOPIC_KEYWORDS["network"]):
        return "network"
    return "unknown"

def hard_check_citations_whitelist(answer_obj: Dict[str, Any], kb_evidence_docs) -> list[str]:
    """
    Ensure every citation is exactly one of the retrieved KB evidence chunks.
    """
    errors = []

    allowed: set[Tuple[str, int]] = set()
    for d in kb_evidence_docs:
        src = d.metadata.get("source", "unknown")
        cid = int(d.metadata.get("chunk_id", -1))
        allowed.add((src, cid))

    citations = answer_obj.get("citations", [])
    for c in citations:
        src = c.get("source")
        cid = c.get("chunk_id")
        try:
            cid_int = int(cid)
        except Exception:
            errors.append(f"Invalid chunk_id in citation: {cid}")
            continue

        if (src, cid_int) not in allowed:
            errors.append(f"Citation not in retrieved KB evidence: {(src, cid_int)}")
    return errors

def hard_check_domain_mismatch(question: str, kb_evidence_docs) -> list[str]:
    errors = []
    q_domain = infer_question_domain(question)

    kb_text = "\n".join(d.page_content for d in kb_evidence_docs).lower()

    # Only trigger "tax mismatch" if KB has strong tax signals
    has_tax_signals = _contains_any(kb_text, TOPIC_KEYWORDS["tax_refund"])

    if q_domain != "tax_refund" and has_tax_signals:
        errors.append("KB evidence appears to be about tax/IRS refunds, but the question is not tax-related.")

    return errors

def hard_check_action_grounding(question: str, answer: str, kb_evidence_docs) -> list[str]:
    """
    Lightweight action grounding: if answer recommends escalation for a refund,
    require KB evidence to mention escalation for refunds (not unrelated contexts).
    """
    errors = []
    ans = answer.lower()
    if "escalat" in ans:
        kb_text = "\n".join(d.page_content for d in kb_evidence_docs).lower()
        # require escalation keyword to appear in KB evidence at all
        if "escalat" not in kb_text:
            errors.append("Answer recommends escalation but KB evidence does not mention escalation.")
    return errors
