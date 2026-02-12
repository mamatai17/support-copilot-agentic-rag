# scripts/openai_create_finetune_job.py
from __future__ import annotations
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def main():
    client = OpenAI()

    training_file_id = os.environ["TRAIN_FILE_ID"]
    validation_file_id = os.environ["VAL_FILE_ID"]

    # Put a fine-tunable base model here (see notes below)
    base_model = os.getenv("FT_BASE_MODEL", "gpt-4.1-nano")

    job = client.fine_tuning.jobs.create(
        model=base_model,
        training_file=training_file_id,
        validation_file=validation_file_id,
        method={
            "type": "supervised",
            "supervised": {"hyperparameters": {"n_epochs": 2}},
        },
    )

    print("JOB_ID:", job.id)
    print("BASE_MODEL:", base_model)

if __name__ == "__main__":
    main()
