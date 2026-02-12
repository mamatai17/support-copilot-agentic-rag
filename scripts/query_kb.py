from rag.embeddings import get_embeddings_model
from rag.vectorstore import load_faiss_index

def main():
    embeddings = get_embeddings_model()
    idx = load_faiss_index("data/index/kb_faiss", embeddings)

    q = "What is a deployment in Kubernetes?"
    hits = idx.similarity_search(q, k=3)

    for i, d in enumerate(hits, 1):
        print(f"\n--- KB HIT {i} ---")
        print("meta:", d.metadata)
        print(d.page_content[:400], "...")
        
if __name__ == "__main__":
    main()
