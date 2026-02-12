# scripts/download_kb_amazonqa_subset.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
from datasets import load_dataset

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATE_DATASETS = [
    "sentence-transformers/amazon-qa",   # common Amazon QA mirror
    "embedding-data/Amazon-QA",          # large QA-style dataset
]

QUESTION_CANDIDATES = ["question", "query", "input", "text", "sent0", "sentence1", "prompt"]
ANSWER_CANDIDATES = ["answer", "output", "response", "label", "sent1", "sentence2", "completion"]


def pick_column(cols: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def main(n: int = 20000):
    last_err = None
    for ds_name in CANDIDATE_DATASETS:
        try:
            ds = load_dataset(ds_name)  # pulls from HF hub
            split = ds["train"] if "train" in ds else next(iter(ds.values()))
            df = split.select(range(min(n, len(split)))).to_pandas()

            cols = df.columns.tolist()
            q_col = pick_column(cols, QUESTION_CANDIDATES)
            a_col = pick_column(cols, ANSWER_CANDIDATES)

            # If we can't find obvious Q/A columns, try a heuristic:
            if q_col is None or a_col is None:
                # pick two text-like columns
                text_cols = [c for c in cols if df[c].dtype == "object"]
                if len(text_cols) >= 2:
                    q_col = q_col or text_cols[0]
                    a_col = a_col or text_cols[1]
                else:
                    raise ValueError(f"Could not infer question/answer columns from: {cols}")

            out = pd.DataFrame({
                "text": ("Q: " + df[q_col].astype(str) + "\nA: " + df[a_col].astype(str)),
                "source": ds_name,
            }).dropna()

            out_path = OUT / "kb_amazonqa.parquet"
            out.to_parquet(out_path, index=False)
            print(f"Loaded dataset: {ds_name}")
            print(f"Using columns: question={q_col} answer={a_col}")
            print(f"Saved {len(out)} KB rows -> {out_path}")
            return

        except Exception as e:
            last_err = e
            print(f"Failed to load {ds_name}: {e}")

    raise RuntimeError(f"All candidate datasets failed. Last error: {last_err}")


if __name__ == "__main__":
    main()
