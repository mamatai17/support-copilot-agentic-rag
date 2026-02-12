import pandas as pd
from langchain_core.documents import Document

def load_kb_passages(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    return df

def kb_to_documents(df: pd.DataFrame, limit: int | None = None) -> list[Document]:
    if limit is not None:
        df = df.head(limit)

    docs = []
    for i, row in df.iterrows():
        docs.append(
            Document(
                page_content=row["text"],
                metadata={
                    "source": row.get("source", "kb"),
                    "chunk_id": int(i),
                    "type": "kb_passage",
                },
            )
        )
    return docs
