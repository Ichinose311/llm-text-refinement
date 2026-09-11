import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def build_text(x):
    return (
        "ユーザ要約:\n"
        f"{x.get('dialogue_summary', '')}\n\n"
        "観光地情報:\n"
        f"{x.get('candidate_information', '')}\n\n"
        "既存推薦文:\n"
        f"{x.get('item_recommendation_sentence', '')}\n\n"
        "v16推薦理由:\n"
        f"{x.get('v16_reason_sentence', '')}"
    )


class RankingDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        x = self.rows[idx]
        enc = self.tokenizer(
            build_text(x),
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        enc["labels"] = float(x["score"])
        return enc


def load_rows(data_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.load(f))
    return rows


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.squeeze(preds)
    labels = np.asarray(labels)
    mse = np.mean((preds - labels) ** 2)
    return {"mse": float(mse)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="llm-jp/llm-jp-modernbert-base")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.data_dir)
    random.shuffle(rows)

    if args.max_train_items is not None:
        rows = rows[: args.max_train_items]

    n_valid = max(1, int(len(rows) * args.valid_ratio))
    valid_rows = rows[:n_valid]
    train_rows = rows[n_valid:]

    print("loaded rows:", len(rows))
    print("train:", len(train_rows))
    print("valid:", len(valid_rows))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="regression",
    )

    train_ds = RankingDataset(train_rows, tokenizer, args.max_length)
    valid_ds = RankingDataset(valid_rows, tokenizer, args.max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mse",
        greater_is_better=False,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    metrics = trainer.evaluate()
    print("eval metrics:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("saved model to:", args.output_dir)


if __name__ == "__main__":
    main()
