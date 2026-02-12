# scripts/openai_check_finetune_job.py
from __future__ import annotations
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def main():
    client = OpenAI()
    job_id = os.environ["JOB_ID"]

    job = client.fine_tuning.jobs.retrieve(job_id)
    print("status:", job.status)
    print("fine_tuned_model:", getattr(job, "fine_tuned_model", None))
    print("error:", getattr(job, "error", None))

if __name__ == "__main__":
    main()
