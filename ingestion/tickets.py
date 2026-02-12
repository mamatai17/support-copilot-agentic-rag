# ingestion/tickets.py
from __future__ import annotations
import pandas as pd
from langchain_core.documents import Document

def load_ticket_pairs(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # normalize
    df = df.dropna(subset=["input", "output"]).reset_index(drop=True)
    df["input"] = df["input"].astype(str)
    df["output"] = df["output"].astype(str)
    return df

def tickets_to_documents(df: pd.DataFrame, limit: int | None = None) -> list[Document]:
    if limit is not None:
        df = df.head(limit)

    docs: list[Document] = []
    for i, row in df.iterrows():
        # Put both the issue and resolution in the stored text
        content = (
            f"Customer: {row['input']}\n"
            f"Support: {row['output']}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "twitter_support_subset",
                    "row_id": int(i),
                    "type": "ticket_case",
                },
            )
        )
    return docs
