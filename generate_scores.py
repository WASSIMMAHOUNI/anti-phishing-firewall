"""
Regenerate data/processed/transformer_scores.csv from data/processed/emails_clean.csv
using the already-trained DistilBERT model (val and test splits only; train excluded).
"""

import pandas as pd
import torch
from text_agent import _load_artifacts, MAX_LENGTH

def main():
    df = pd.read_csv("data/processed/emails_clean.csv")
    eval_df = df[df["split"].isin(["val", "test"])].copy().reset_index(drop=True)

    tokenizer, model, device = _load_artifacts()
    texts = (
        eval_df["subject"].fillna("").astype(str) + " " + eval_df["body"].fillna("").astype(str)
    ).str.strip().tolist()

    scores = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
        scores.extend(probs)

    out = pd.DataFrame({
        "email_id": eval_df["email_id"].values,
        "split": eval_df["split"].values,
        "text_score": scores,
    })

    assert len(out) == 14683, f"Expected 14683 rows, got {len(out)}"
    assert (out["split"] == "val").sum() == 7340
    assert (out["split"] == "test").sum() == 7343
    assert not out.isna().any().any()
    assert out["email_id"].is_unique

    out.to_csv("data/processed/transformer_scores.csv", index=False)
    print("Saved data/processed/transformer_scores.csv:", out.shape)

if __name__ == "__main__":
    main()
