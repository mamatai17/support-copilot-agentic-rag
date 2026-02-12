from __future__ import annotations
from pathlib import Path
import json
import random
import pandas as pd

from rag.embeddings import get_embeddings_model
from rag.vectorstore import load_faiss_index

DATA_OUT = Path("finetune/data")
DATA_OUT.mkdir(parents=True, exist_ok=True)

def format_kb_for_training(docs) -> str:
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        cid = d.metadata.get("chunk_id", -1)
        lines.append(f"CITE_KEY=({src},{cid})\n{d.page_content}")
    return "\n\n".join(lines)

def main(n: int = 200, seed: int = 7):
    random.seed(seed)

    df = pd.read_parquet("data/raw/twitter_support_subset.parquet").dropna()
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    embeddings = get_embeddings_model()
    kb = load_faiss_index("data/index/kb_faiss", embeddings)

    rows = []
    for i, r in df.iterrows():
        question = str(r["input"]).strip()
        support_reply = str(r["output"]).strip()

        kb_docs = kb.similarity_search(question, k=5)
        kb_text = format_kb_for_training(kb_docs)

        # Heuristic confidence: if KB has "refund" and question has "refund", medium; else low.
        q_low = question.lower()
        kb_low = kb_text.lower()
        same_topic = ("refund" in q_low and "refund" in kb_low) or ("return" in q_low and "return" in kb_low)
        confidence = "medium" if same_topic else "low"

        # For low confidence, we encourage asking for missing info.
        missing_info = None
        if confidence == "low":
            missing_info = "The knowledge base doesn't clearly cover this exact case. Need order details and refund status/timeline."

        # We’ll keep citations simple for now: cite the top-1 KB chunk if medium, else none.
        citations = []
        if confidence == "medium" and kb_docs:
            citations = [{"source": kb_docs[0].metadata["source"], "chunk_id": int(kb_docs[0].metadata["chunk_id"])}]

        # Convert the support reply into next steps (simple heuristic)
        next_steps = []
        if "dm" in support_reply.lower() or "direct message" in support_reply.lower():
            next_steps.append("Contact support via direct message or the official support channel.")
        if "bank" in support_reply.lower():
            next_steps.append("Check with your bank for pending refunds or reversals.")
        if not next_steps:
            next_steps = ["Contact support with your order details.", "Share the refund status/timeline if available."]

        ideal_json = {
            "answer": support_reply,
            "citations": citations,
            "confidence": confidence,
            "missing_info": missing_info,
            "next_steps": next_steps,
            "similar_cases": []
        }

        # Training record: (system + user) -> assistant JSON
        rows.append({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a support copilot. Use ONLY KB_EVIDENCE for factual/policy claims. "
                        "Return valid JSON matching the RagAnswer schema."
                    )
                },
                {
                    "role": "user",
                    "content": f"QUESTION:\n{question}\n\nKB_EVIDENCE:\n{kb_text}"
                },
                {
                    "role": "assistant",
                    "content": json.dumps(ideal_json, ensure_ascii=False)
                }
            ]
        })

    out_path = DATA_OUT / "train.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} examples to {out_path}")

if __name__ == "__main__":
    main()
