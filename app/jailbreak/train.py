"""
Fine-tunes DistilBERT for binary jailbreak classification.

Train/val split: stratified 90/10 from train_set.csv
Test: test_set_clean.csv — mixed labels (JBB jailbreak + held-out Alpaca benign),
touched only once, at the end, for final reported metrics.
"""

import os
os.environ["HF_DATASETS_CACHE"] = "./data/raw_cache"

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 96
BATCH_SIZE = 128
OUTPUT_DIR = "./models/jailbreak_classifier"


def load_and_split_data():
    train_df = pd.read_csv("./data/train_set.csv")
    val_df = pd.read_csv("./data/val_set_clean.csv")
    test_df = pd.read_csv("./data/test_set_clean.csv")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert "label" in df.columns, f"{name}_df missing 'label' column"
        assert df["label"].nunique() == 2, (
            f"{name} set has only one class: {df['label'].unique()}. "
            f"Check contamination filtering didn't strip an entire label."
        )

    print(f"Train: {len(train_df)} rows, label dist:\n{train_df['label'].value_counts()}")
    print(f"Val:   {len(val_df)} rows, label dist:\n{val_df['label'].value_counts()}")
    print(f"Test:  {len(test_df)} rows, label dist:\n{test_df['label'].value_counts()}")

    return train_df, val_df, test_df


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    torch.set_num_threads(8)
    print("Running pre-training label bias gate check against data/train_set.csv...")
    from scripts.check_label_bias import check_label_bias
    if not check_label_bias("./data/train_set.csv"):
        import sys
        print("[FATAL] Pre-training bias check FAILED! Aborting training.", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df, val_df, test_df = load_and_split_data()


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    keep_cols = ["input_ids", "attention_mask", "label"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep_cols])

    train_ds.set_format(type="torch", columns=keep_cols)
    val_ds.set_format(type="torch", columns=keep_cols)
    test_ds.set_format(type="torch", columns=keep_cols)

    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        fp16=(device == "cuda"),
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating on held-out test set (JBB + Alpaca-heldout, touched once, final numbers)...")
    test_results = trainer.evaluate(eval_dataset=test_ds)
    print(f"\nFinal test metrics: {test_results}")

    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(f"{OUTPUT_DIR}/test_metrics.txt", "w") as f:
        f.write(str(test_results))

    print("\nDone. Model + tokenizer saved. Test metrics written to test_metrics.txt")


if __name__ == "__main__":
    main()