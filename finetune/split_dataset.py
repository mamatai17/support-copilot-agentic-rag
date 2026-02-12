# finetune/split_dataset.py
from __future__ import annotations
from pathlib import Path
import random

def main(seed: int = 7, val_ratio: float = 0.15):
    random.seed(seed)
    src = Path("finetune/data/train.jsonl")
    lines = src.read_text(encoding="utf-8").splitlines()
    random.shuffle(lines)

    n_val = int(len(lines) * val_ratio)
    val = lines[:n_val]
    train = lines[n_val:]

    out_dir = Path("finetune/data")
    (out_dir / "train_split.jsonl").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out_dir / "val_split.jsonl").write_text("\n".join(val) + "\n", encoding="utf-8")

    print(f"train={len(train)} val={len(val)}")

if __name__ == "__main__":
    main()
