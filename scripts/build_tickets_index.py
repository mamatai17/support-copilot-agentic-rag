# scripts/build_tickets_index.py
from pathlib import Path
from rag.embeddings import get_embeddings_model
from rag.vectorstore import build_faiss_index
from ingestion.tickets import load_ticket_pairs, tickets_to_documents

INDEX_DIR = Path("data/index/tickets_faiss")

def main(limit: int = 5000):
    df = load_ticket_pairs("data/raw/twitter_support_subset.parquet")
    docs = tickets_to_documents(df, limit=limit)

    embeddings = get_embeddings_model()
    index = build_faiss_index(docs, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(INDEX_DIR))
    print(f"Saved tickets index -> {INDEX_DIR} (docs={len(docs)})")

if __name__ == "__main__":
    main()
