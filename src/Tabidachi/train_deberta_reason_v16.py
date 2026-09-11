import argparse
import glob
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments


MODEL_NAME = "globis-university/deberta-v3-japanese-large"


class RecommendationDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sep = self.tokenizer.sep_token

        text = (
            f"{item['dialogue_summary']}"
            f"{sep}{item['recommendation_sentence']}"
            f"{sep}{item['candidate_information']}"
        )

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(float(item["score"]), dtype=torch.float),
        }


class DebertaRegressionModel(nn.Module):
    def __init__(self, pretrained_model_name: str):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(pretrained_model_name)
        self.regression_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.regression_head(cls_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def load_json_data(data_dir: str) -> List[Dict]:
    all_data = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        all_data.extend(rows)
    return all_data


def split_data(data: List[Dict], valid_ratio: float, seed: int):
    data = list(data)
    random.Random(seed).shuffle(data)

    if len(data) < 2:
        raise ValueError("Need at least 2 examples for train/valid split.")

    valid_size = max(1, int(len(data) * valid_ratio))
    valid_size = min(valid_size, len(data) - 1)

    valid_data = data[:valid_size]
    train_data = data[valid_size:]
    return train_data, valid_data


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}

def main():
    parser = argparse.ArgumentParser(
        description="Train clean 3-input DeBERTa evaluator for v16 recommendation reasons."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_json_data(args.data_dir)
    if args.max_train_items is not None:
        rows = rows[: args.max_train_items]

    print(f"loaded examples: {len(rows)}")
    positives = sum(1 for x in rows if int(x["score"]) == 1)
    negatives = len(rows) - positives
    print(f"label counts: positive={positives}, negative={negatives}")

    train_rows, valid_rows = split_data(rows, args.valid_ratio, args.seed)
    print(f"train={len(train_rows)}, valid={len(valid_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = RecommendationDataset(train_rows, tokenizer, max_length=args.max_length)
    valid_dataset = RecommendationDataset(valid_rows, tokenizer, max_length=args.max_length)

    model = DebertaRegressionModel(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        report_to="none",
        save_safetensors=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    print("eval metrics:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved model to: {args.output_dir}")


if __name__ == "__main__":
    main()
