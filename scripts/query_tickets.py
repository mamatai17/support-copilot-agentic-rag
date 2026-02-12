# scripts/query_tickets.py
from rag.embeddings import get_embeddings_model
from rag.vectorstore import load_faiss_index

def main():
    embeddings = get_embeddings_model()
    idx = load_faiss_index("data/index/tickets_faiss", embeddings)

    q = "My internet shows full bars but nothing loads"
    hits = idx.similarity_search(q, k=3)

    for i, d in enumerate(hits, 1):
        print(f"\n--- HIT {i} ---")
        print("meta:", d.metadata)
        print(d.page_content)

if __name__ == "__main__":
    main()
