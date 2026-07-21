"""
Builds train/val/test data for the jailbreak classifier.

- Train: rubend18 + TrustAIRLab (label=1) + Alpaca-train (label=0)
- Val:   JBB slice A (label=1) + Alpaca-val slice (label=0)  -- cross-source, same as test
- Test:  JBB slice B (label=1) + Alpaca-test slice (label=0) -- cross-source, disjoint from val

Val and test are both held out from JBB/Alpaca and disjoint from each other and from train.
This is what makes val an actual generalization signal instead of a same-distribution echo.
"""

import os
os.environ["HF_DATASETS_CACHE"] = "./data/raw_cache"

from datasets import load_dataset
import pandas as pd


JAILBREAK_SCORE_THRESHOLD = 50
ALPACA_VAL_HOLDOUT = 100
ALPACA_TEST_HOLDOUT = 150
JBB_VAL_FRACTION = 0.33  # ~100 of JBB's 300 rows go to val, rest to test


def load_jailbreak_examples_rubend18() -> pd.DataFrame:
    ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", cache_dir="./data/raw_cache")
    raw_df = ds["train"].to_pandas()
    raw_df["Jailbreak Score"] = pd.to_numeric(raw_df["Jailbreak Score"], errors="coerce")
    filtered_df = raw_df[raw_df["Jailbreak Score"] > JAILBREAK_SCORE_THRESHOLD].copy()
    print(f"rubend18: {len(raw_df)} rows -> {len(filtered_df)} after score filter")
    return pd.DataFrame({
        "text": filtered_df["Prompt"].astype(str).values,
        "label": 1,
        "source": "rubend18_jailbreak_prompts",
    })


def load_jailbreak_examples_wild() -> pd.DataFrame:
    ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", cache_dir="./data/raw_cache")
    raw_df = ds["train"].to_pandas()
    print(f"TrustAIRLab: {len(raw_df)} rows")
    return pd.DataFrame({
        "text": raw_df["prompt"].astype(str).values,
        "label": 1,
        "source": "trustairlab_in_the_wild",
    })


def load_benign_examples_split(n_train: int, n_val: int, n_test: int, random_state: int = 42):
    """Three-way disjoint split of Alpaca: train / val / test."""
    ds = load_dataset("tatsu-lab/alpaca", cache_dir="./data/raw_cache")
    raw_df = ds["train"].to_pandas()

    text_col = "instruction"
    if text_col not in raw_df.columns:
        raise KeyError(f"Expected '{text_col}' not in Alpaca columns: {raw_df.columns.tolist()}")

    total_needed = n_train + n_val + n_test
    if total_needed > len(raw_df):
        raise ValueError(f"Requested {total_needed} rows but Alpaca only has {len(raw_df)}")

    sampled = raw_df.sample(n=total_needed, random_state=random_state).reset_index(drop=True)

    train_slice = sampled.iloc[:n_train]
    val_slice = sampled.iloc[n_train:n_train + n_val]
    test_slice = sampled.iloc[n_train + n_val:n_train + n_val + n_test]

    def to_df(slice_df, source_tag):
        return pd.DataFrame({
            "text": slice_df[text_col].astype(str).values,
            "label": 0,
            "source": source_tag,
        })

    return (
        to_df(train_slice, "tatsu_lab_alpaca_train"),
        to_df(val_slice, "tatsu_lab_alpaca_val"),
        to_df(test_slice, "tatsu_lab_alpaca_test"),
    )


def load_jbb_split(random_state: int = 42):
    """Split JBB into val and test slices — disjoint, same distribution."""
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison", cache_dir="./data/raw_cache")
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    raw_df = ds[split_name].to_pandas()

    shuffled = raw_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_val = int(len(shuffled) * JBB_VAL_FRACTION)

    val_slice = shuffled.iloc[:n_val]
    test_slice = shuffled.iloc[n_val:]

    def to_df(slice_df, source_tag):
        return pd.DataFrame({
            "text": slice_df["prompt"].astype(str).values,
            "label": 1,
            "source": source_tag,
        })

    return to_df(val_slice, "jailbreakbench_val"), to_df(test_slice, "jailbreakbench_test")


if __name__ == "__main__":
    print("Building jailbreak training examples...")
    jb_rubend18 = load_jailbreak_examples_rubend18()
    jb_wild = load_jailbreak_examples_wild()
    jb_train = pd.concat([jb_rubend18, jb_wild], ignore_index=True)
    jb_train = jb_train.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    print(f"Combined jailbreak training examples: {len(jb_train)}")

    n_benign_train = len(jb_train) * 2

    print(f"\nLoading Alpaca 3-way split: {n_benign_train} train / {ALPACA_VAL_HOLDOUT} val / {ALPACA_TEST_HOLDOUT} test...")
    alpaca_train, alpaca_val, alpaca_test = load_benign_examples_split(
        n_train=n_benign_train, n_val=ALPACA_VAL_HOLDOUT, n_test=ALPACA_TEST_HOLDOUT,
    )

    print("\nLoading JBB val/test split...")
    jbb_val, jbb_test = load_jbb_split()

    # --- Train ---
    train_df = pd.concat([jb_train, alpaca_train], ignore_index=True)
    train_df = train_df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nFinal train set: {len(train_df)} rows\n{train_df['label'].value_counts()}")
    train_df.to_csv("./data/train_set.csv", index=False)

    # --- Val (cross-source, same distribution as test) ---
    val_df = pd.concat([jbb_val, alpaca_val], ignore_index=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nFinal val set: {len(val_df)} rows\n{val_df['label'].value_counts()}")
    val_df.to_csv("./data/val_set.csv", index=False)

    # --- Test (cross-source, disjoint from val) ---
    test_df = pd.concat([jbb_test, alpaca_test], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nFinal test set: {len(test_df)} rows\n{test_df['label'].value_counts()}")
    test_df.to_csv("./data/test_set.csv", index=False)

    print("\nSaved train_set.csv, val_set.csv, test_set.csv")
    print("Next: rerun check_contamination.py against BOTH val_set.csv and test_set.csv, then rerun train.py")