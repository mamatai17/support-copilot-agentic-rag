# scripts/hello_retrieval.py
from langchain_core.documents import Document
from rag.embeddings import get_embeddings_model
from rag.vectorstore import build_faiss_index, query_index

def main():
    docs = [
        Document(page_content="Refunds take 5-10 business days. EU refunds may take longer.", metadata={"source":"policy.md", "chunk_id": 0}),
        Document(page_content="If a user is locked out, advise password reset and verify email.", metadata={"source":"runbook.md", "chunk_id": 0}),
        Document(page_content="Escalate billing issues if charge is duplicated or pending > 7 days.", metadata={"source":"runbook.md", "chunk_id": 1}),
    ]

    embeddings = get_embeddings_model()
    index = build_faiss_index(docs, embeddings)

    q = "My EU refund hasn't arrived. What should I do?"
    hits = query_index(index, q, k=2)

    for i, d in enumerate(hits, 1):
        print(f"\n--- Hit {i} ---")
        print("source:", d.metadata)
        print(d.page_content)

if __name__ == "__main__":
    main()