# eval/run_eval.py
from __future__ import annotations
import os
import json
from dotenv import load_dotenv
from collections import Counter
from typing import Any, Dict, List

from agent.graph import build_graph

load_dotenv()

QUESTIONS = [
    # refund-ish
    "My EU refund hasn't arrived. What should I do?",
    "I asked for a refund a week ago. Still nothing. What next?",
    # connectivity-ish
    "Full bars LTE but nothing loads. Help!",
    "My data is very slow even with good signal.",
    # billing-ish
    "I was charged twice. What should I do?",
]

def run_once(model_name: str) -> List[Dict[str, Any]]:
    os.environ["GEN_MODEL"] = model_name
    graph = build_graph()

    results = []
    for q in QUESTIONS:
        state = {
            "question": q,
            "query": q,
            "kb_k": 5,
            "tickets_k": 3,
            "kb_evidence": [],
            "ticket_evidence": [],
            "answer": None,
            "decision": None,
            "validator_feedback": None,
            "retries": 0,
        }
        out = graph.invoke(state)

        ans = out["answer"].model_dump() if out.get("answer") else None
        results.append({
            "question": q,
            "decision": out.get("decision"),
            "feedback": out.get("validator_feedback"),
            "confidence": ans.get("confidence") if ans else None,
            "num_citations": len(ans.get("citations", [])) if ans else 0,
            "num_similar_cases": len(ans.get("similar_cases", [])) if ans else 0,
        })
    return results

def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    decisions = Counter(r["decision"] for r in results)
    confs = Counter(r["confidence"] for r in results)
    return {
        "decisions": dict(decisions),
        "confidence": dict(confs),
        "avg_citations": sum(r["num_citations"] for r in results) / max(1, len(results)),
        "avg_similar_cases": sum(r["num_similar_cases"] for r in results) / max(1, len(results)),
        "retry_rate": decisions.get("RETRY_WITH_MORE_CONTEXT", 0) / max(1, sum(decisions.values())),
    }

def main():
    base = os.getenv("BASE_MODEL", "gpt-4o-mini")
    tuned = os.getenv("TUNED_MODEL", "")

    print("BASE_MODEL =", base)
    base_res = run_once(base)
    print("\nBASE SUMMARY:\n", json.dumps(summarize(base_res), indent=2))
    print("\nBASE SAMPLE:\n", json.dumps(base_res[:2], indent=2))

    if tuned:
        print("\nTUNED_MODEL =", tuned)
        tuned_res = run_once(tuned)
        print("\nTUNED SUMMARY:\n", json.dumps(summarize(tuned_res), indent=2))
        print("\nTUNED SAMPLE:\n", json.dumps(tuned_res[:2], indent=2))
    else:
        print("\nSet TUNED_MODEL env var when the fine-tuned model is ready.")

if __name__ == "__main__":
    main()
