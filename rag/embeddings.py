# rag/embeddings.py
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings

def get_embeddings_model():
    # A stable default for embeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")
