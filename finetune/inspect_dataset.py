import json
from pathlib import Path

def main():
    p = Path("finetune/data/train.jsonl")
    with p.open("r", encoding="utf-8") as f:
        first = json.loads(next(f))
    print(json.dumps(first, indent=2)[:2000])

if __name__ == "__main__":
    main()
