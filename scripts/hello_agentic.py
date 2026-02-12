# scripts/hello_agentic.py
from agent.graph import build_graph

def main():
    graph = build_graph()

    question = "My EU refund hasn't arrived. What should I do?"
    init_state = {
        "question": question,
        "query": question,
        "kb_k": 5,
        "tickets_k": 3,
        "kb_evidence": [],
        "ticket_evidence": [],
        "answer": None,
        "decision": None,
        "validator_feedback": None,
        "retries": 0,
    }

    out = graph.invoke(init_state)

    print("\nDECISION:", out.get("decision"))
    print("VALIDATOR_FEEDBACK:", out.get("validator_feedback"))
    if out.get("answer"):
        print("\nANSWER JSON:\n", out["answer"].model_dump_json(indent=2))

if __name__ == "__main__":
    main()
