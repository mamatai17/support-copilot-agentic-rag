from pathlib import Path
import pandas as pd
from datasets import load_dataset

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

def main(n: int = 5000):
    ds = load_dataset("MohammadOthman/mo-customer-support-tweets-945k")  # public dataset :contentReference[oaicite:2]{index=2}
    split = ds["train"]

    # Take a subset for fast iteration
    df = split.select(range(min(n, len(split)))).to_pandas()

    # Save locally (gitignored)
    out_path = OUT / "twitter_support_subset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    main()
