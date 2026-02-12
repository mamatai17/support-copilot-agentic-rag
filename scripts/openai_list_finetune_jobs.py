# scripts/openai_list_finetune_jobs.py
from __future__ import annotations
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    client = OpenAI()
    jobs = client.fine_tuning.jobs.list(limit=20)

    print("Recent fine-tuning jobs:")
    for j in jobs.data:
        print("-", j.id, j.status, "model=", j.model, "fine_tuned_model=", getattr(j, "fine_tuned_model", None))

if __name__ == "__main__":
    main()
