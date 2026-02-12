# scripts/openai_upload_finetune_files.py
from __future__ import annotations
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    client = OpenAI()

    train_file = client.files.create(
        file=open("finetune/data/train_split.jsonl", "rb"),
        purpose="fine-tune",
    )
    val_file = client.files.create(
        file=open("finetune/data/val_split.jsonl", "rb"),
        purpose="fine-tune",
    )

    print("TRAIN_FILE_ID:", train_file.id)
    print("VAL_FILE_ID:", val_file.id)

if __name__ == "__main__":
    main()
