# rag/validate.py
from rag.schema import RagAnswer

def validate_answer(ans: RagAnswer, evidence_docs) -> list[str]:
    errors = []
    allowed = set()
    for d in evidence_docs:
        allowed.add((d.metadata.get("source"), int(d.metadata.get("chunk_id"))))

    for c in ans.citations:
        if (c.source, c.chunk_id) not in allowed:
            errors.append(f"Invalid citation: {(c.source, c.chunk_id)} not in retrieved evidence.")

    if ans.confidence not in {"high", "medium", "low"}:
        errors.append("confidence must be one of: high, medium, low.")

    if not isinstance(ans.next_steps, list) or any(not isinstance(x, str) for x in ans.next_steps):
        errors.append("next_steps must be a list of strings.")

    return errors
