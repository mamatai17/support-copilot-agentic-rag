from pathlib import Path
from rag.embeddings import get_embeddings_model
from rag.vectorstore import build_faiss_index
from ingestion.kb_passages import load_kb_passages, kb_to_documents

INDEX_DIR = Path("data/index/kb_faiss")

def main(limit: int = 20000):
    df = load_kb_passages("data/raw/kb_amazonqa.parquet")
    docs = kb_to_documents(df, limit=limit)

    embeddings = get_embeddings_model()
    index = build_faiss_index(docs, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(INDEX_DIR))
    print(f"Saved KB index -> {INDEX_DIR} (docs={len(docs)})")

if __name__ == "__main__":
    main()
