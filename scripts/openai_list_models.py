# scripts/openai_list_models.py
from __future__ import annotations
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    client = OpenAI()
    models = client.models.list()

    # Print model IDs (filtering helps readability)
    ids = sorted([m.id for m in models.data])

    print("Total models visible:", len(ids))
    print("\n--- Models (first 200) ---")
    for mid in ids[:200]:
        print(mid)

    # Convenience: show anything that looks like a candidate
    print("\n--- Candidates containing 'ft' or 'fine' or 'gpt-3.5' or 'gpt-4.1' ---")
    for mid in ids:
        low = mid.lower()
        if any(x in low for x in ["ft", "fine", "gpt-3.5", "gpt-4.1", "gpt-4o"]):
            print(mid)

if __name__ == "__main__":
    main()
