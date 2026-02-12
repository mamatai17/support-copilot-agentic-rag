# rag/vectorstore.py
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def build_faiss_index(docs: list[Document], embeddings):
    return FAISS.from_documents(docs, embeddings)

def load_faiss_index(path: str, embeddings):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def query_index(index: FAISS, query: str, k: int = 5):
    return index.similarity_search(query, k=k)
