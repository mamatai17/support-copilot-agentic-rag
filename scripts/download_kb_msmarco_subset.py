from pathlib import Path
import pandas as pd
from datasets import load_dataset

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

def main(n: int = 5000):
    ds = load_dataset("microsoft/ms_marco", "v1.1")  # public retrieval dataset
    # We'll use the "train" split passages
    split = ds["train"]

    # The schema includes query/answers and passage candidates; we’ll extract passages
    # To keep it simple, we take rows and pull their passages field into a flat table.
    rows = []
    for ex in split.select(range(min(n, len(split)))):
        passages = ex.get("passages", {})
        texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])
        for j, t in enumerate(texts[:5]):  # cap per example
            rows.append({
                "text": t,
                "is_selected": int(is_selected[j]) if j < len(is_selected) else 0,
                "source": "msmarco",
            })

    df = pd.DataFrame(rows).dropna()
    out_path = OUT / "kb_msmarco_passages.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} passages -> {out_path}")

if __name__ == "__main__":
    main()
