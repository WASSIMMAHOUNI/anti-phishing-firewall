import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextAgent:
    def __init__(self, model_dir="models/distilbert_phishing_model", max_length=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def score_email(self, subject, body):
        text = f"{subject or ""} {body or ""}".strip()
        enc = self.tokenizer(text, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            prob = torch.softmax(self.model(**enc).logits, dim=-1)[0, 1].item()
        return float(prob)

if __name__ == "__main__":
    import pandas as pd, numpy as np
    agent = TextAgent()
    print("Device:", agent.device)
    print("Normal:", round(agent.score_email("Project sync", "Hi team, meet tomorrow at 10 AM in room B2."), 4))
    print("Phish: ", round(agent.score_email("URGENT: Account Suspended", "Click here now to verify your password."), 4))
    df = pd.read_csv("data/processed/emails_clean.csv")
    val = df[df["split"] == "val"].head(100)
    _ = agent.score_email(str(val.iloc[0]["subject"]), str(val.iloc[0]["body"]))
    times = []
    for _, r in val.iterrows():
        t0 = time.perf_counter()
        _ = agent.score_email("" if pd.isna(r["subject"]) else str(r["subject"]), "" if pd.isna(r["body"]) else str(r["body"]))
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    print(f"Mean: {arr.mean():.2f} ms | Median: {np.median(arr):.2f} ms | p95: {np.percentile(arr, 95):.2f} ms | Throughput: {1000.0/arr.mean():.1f} emails/s")
