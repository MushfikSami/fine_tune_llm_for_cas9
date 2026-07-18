#!/usr/bin/env python
"""Step 3: LoRA fine-tune DNABERT-2 as an on-target efficiency regressor.

Loads the DNABERT-2-117M sequence-classification model with a single-output
regression head, wraps it with LoRA (PEFT) to keep VRAM low, and trains on the
Doench/Azimuth 2016 efficiency dataset produced by ``02_preprocess_dnabert.py``.
The trained LoRA adapter is saved to ``models/dnabert_lora/``.
"""
from __future__ import annotations

import argparse

import numpy as np
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from scipy.stats import pearsonr, spearmanr
from transformers import (DataCollatorWithPadding, Trainer, TrainingArguments)

from dnabert_utils import (LORA_DIR, LORA_TARGET_MODULES, PROCESSED_DIR,
                           load_dnabert)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # DNABERT-2 may return predictions as a tuple (logits, ...); take the logits.
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = np.asarray(preds).squeeze()
    labels = np.asarray(labels).squeeze()
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    spear = float(spearmanr(preds, labels).correlation)
    pear = float(pearsonr(preds, labels)[0])
    return {"rmse": rmse, "spearman": spear, "pearson": pear}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap training steps (for a quick smoke test)")
    args = ap.parse_args()

    data_dir = PROCESSED_DIR / "efficiency"
    if not (data_dir / "dataset_dict.json").exists():
        raise SystemExit(f"Missing {data_dir}. Run 02_preprocess_dnabert.py first.")
    ds = load_from_disk(str(data_dir))
    keep = {"input_ids", "attention_mask", "label"}
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    tokenizer, model = load_dnabert(num_labels=1)

    lora_cfg = LoraConfig(
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=["regressor"],  # train the fresh regression head fully
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorWithPadding(tokenizer)
    training_args = TrainingArguments(
        output_dir=str(LORA_DIR / "_trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\nFinal eval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    LORA_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(LORA_DIR))
    tokenizer.save_pretrained(str(LORA_DIR))
    print(f"\nSaved LoRA adapter to {LORA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
