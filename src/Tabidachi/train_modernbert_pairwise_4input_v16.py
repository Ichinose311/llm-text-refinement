import argparse
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
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


def load_groups(data_dir):
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            for x in json.load(f):
                groups[(x["source_file"], x["data_id"])].append(x)
    return groups


def make_pairs(group_items, max_negatives_per_positive=None):
    pairs = []
    for _, items in group_items:
        positives = [x for x in items if int(x["score"]) > 0]
        negatives = [x for x in items if int(x["score"]) <= 0]
        if not positives or not negatives:
            continue

        for pos in positives:
            negs = list(negatives)
            random.shuffle(negs)
            if max_negatives_per_positive is not None:
                negs = negs[:max_negatives_per_positive]
            for neg in negs:
                pairs.append((pos, neg))
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def encode(self, x):
        return self.tokenizer(
            build_text(x),
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        )

    def __getitem__(self, idx):
        pos, neg = self.pairs[idx]
        return {
            "pos": self.encode(pos),
            "neg": self.encode(neg),
        }


class PairwiseCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        pos = [x["pos"] for x in features]
        neg = [x["neg"] for x in features]

        pos_batch = self.tokenizer.pad(pos, padding=True, return_tensors="pt")
        neg_batch = self.tokenizer.pad(neg, padding=True, return_tensors="pt")

        batch = {}
        for k, v in pos_batch.items():
            batch[f"pos_{k}"] = v
        for k, v in neg_batch.items():
            batch[f"neg_{k}"] = v

        batch["labels"] = torch.ones(len(features), dtype=torch.float)
        return batch


class PairwiseTrainer(Trainer):
    def __init__(self, *args, margin=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("labels", None)

        pos_inputs = {
            k.replace("pos_", ""): v
            for k, v in inputs.items()
            if k.startswith("pos_")
        }
        neg_inputs = {
            k.replace("neg_", ""): v
            for k, v in inputs.items()
            if k.startswith("neg_")
        }

        pos_scores = model(**pos_inputs).logits.squeeze(-1)
        neg_scores = model(**neg_inputs).logits.squeeze(-1)

        diff = pos_scores - neg_scores
        loss = F.softplus(self.margin - diff).mean()

        if return_outputs:
            logits = torch.stack([pos_scores, neg_scores], dim=1)
            return loss, {"logits": logits}
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="llm-jp/llm-jp-modernbert-base")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-negatives-per-positive", type=int, default=None)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    groups = load_groups(args.data_dir)
    group_items = list(groups.items())
    random.shuffle(group_items)

    n_valid = max(1, int(len(group_items) * args.valid_ratio))
    valid_group_items = group_items[:n_valid]
    train_group_items = group_items[n_valid:]

    train_pairs = make_pairs(train_group_items, args.max_negatives_per_positive)
    valid_pairs = make_pairs(valid_group_items, args.max_negatives_per_positive)

    random.shuffle(train_pairs)
    random.shuffle(valid_pairs)

    if args.max_pairs is not None:
        train_pairs = train_pairs[: args.max_pairs]
        valid_pairs = valid_pairs[: max(1, min(len(valid_pairs), args.max_pairs // 5))]

    print("groups:", len(groups))
    print("train groups:", len(train_group_items))
    print("valid groups:", len(valid_group_items))
    print("train pairs:", len(train_pairs))
    print("valid pairs:", len(valid_pairs))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="regression",
    )

    train_ds = PairwiseDataset(train_pairs, tokenizer, args.max_length)
    valid_ds = PairwiseDataset(valid_pairs, tokenizer, args.max_length)
    collator = PairwiseCollator(tokenizer)

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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        margin=args.margin,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    metrics = trainer.evaluate()
    print("eval metrics:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("saved model to:", args.output_dir)


if __name__ == "__main__":
    main()
