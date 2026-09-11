import argparse
import glob
import json
import os
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

try:
    from trl import DPOConfig
except ImportError:
    DPOConfig = None


DEFAULT_MODEL = "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1"


def load_preference_rows(data_dir: str, max_samples: int | None = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    skipped_same = 0

    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            prompt = item["prompt"].rstrip() + "\n\n出力:\n"
            chosen = item["chosen_recommendation_sentence"].strip()
            rejected = item["rejected_recommendation_sentence"].strip()

            if chosen == rejected:
                skipped_same += 1
                continue

            rows.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

            if max_samples is not None and len(rows) >= max_samples:
                print(f"loaded preference pairs: {len(rows)}")
                print(f"skipped same pairs: {skipped_same}")
                return rows

    print(f"loaded preference pairs: {len(rows)}")
    print(f"skipped same pairs: {skipped_same}")
    return rows


def build_training_args(args):
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    common = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=False,
    )

    if DPOConfig is not None:
        return DPOConfig(
            **common,
            beta=args.beta,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
        )

    return TrainingArguments(**common)


def build_trainer(model, tokenizer, train_dataset, peft_config, training_args, args):
    if DPOConfig is not None:
        try:
            return DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                train_dataset=train_dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
            )
        except TypeError:
            return DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                peft_config=peft_config,
            )

    return DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    rows = load_preference_rows(args.data_dir, args.max_samples)
    if not rows:
        raise ValueError("No usable preference pairs found.")

    train_dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = build_training_args(args)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config,
        training_args=training_args,
        args=args,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"saved DPO LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
