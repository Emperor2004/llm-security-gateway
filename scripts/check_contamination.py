"""
Verifies train/test separation by checking cosine similarity between
every test example and every train example. Any test example too similar
to something in train gets flagged and excluded — this is the artifact
that substantiates a "contamination-free" claim in your README.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json

SIMILARITY_THRESHOLD = 0.85  # [Guessing] reasonable default; tune based on your actual distribution


def check_contamination(
    train_csv: str = "./data/train_set.csv",
    test_csv: str = "./data/test_set.csv",
    output_json: str = "./data/contamination_report.json",
    output_csv: str = "./data/test_set_clean.csv",
) -> pd.DataFrame:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Embedding {len(train_df)} train examples...")
    train_emb = embedder.encode(train_df["text"].tolist(), show_progress_bar=True)

    print(f"Embedding {len(test_df)} test examples...")
    test_emb = embedder.encode(test_df["text"].tolist(), show_progress_bar=True)

    train_norm = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)
    test_norm = test_emb / np.linalg.norm(test_emb, axis=1, keepdims=True)
    sim_matrix = np.dot(test_norm, train_norm.T)

    max_sim = sim_matrix.max(axis=1)
    nearest_train_idx = sim_matrix.argmax(axis=1)

    print("\nmax_sim distribution:")
    print(f"  min: {max_sim.min():.4f}")
    print(f"  max: {max_sim.max():.4f}")
    print(f"  mean: {max_sim.mean():.4f}")
    print(f"  median: {np.median(max_sim):.4f}")
    print(f"  95th percentile: {np.percentile(max_sim, 95):.4f}")

    test_df = test_df.copy()
    test_df["max_train_similarity"] = max_sim
    test_df["contaminated"] = max_sim > SIMILARITY_THRESHOLD
    test_df["nearest_train_text"] = train_df["text"].iloc[nearest_train_idx].values

    n_contaminated = test_df["contaminated"].sum()
    report = {
        "source_file": test_csv,
        "total_test_examples": len(test_df),
        "contaminated_count": int(n_contaminated),
        "contaminated_pct": round(100 * n_contaminated / len(test_df), 2),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "clean_test_count": int(len(test_df) - n_contaminated),
    }

    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)

    clean_test_df = test_df[~test_df["contaminated"]].reset_index(drop=True)
    clean_test_df.to_csv(output_csv, index=False)  # was hardcoded before — now uses the parameter

    print(f"\n{report}")
    print(f"\nClean test set saved to {output_csv} ({len(clean_test_df)} rows)")
    print(f"Full report saved to {output_json}")

    return test_df


if __name__ == "__main__":
    import sys
    target_file = sys.argv[1] if len(sys.argv) > 1 else "./data/test_set.csv"
    output_clean = target_file.replace(".csv", "_clean.csv")
    output_report = target_file.replace(".csv", "_contamination_report.json")

    check_contamination(
        train_csv="./data/train_set.csv",
        test_csv=target_file,
        output_json=output_report,
        output_csv=output_clean,
    )