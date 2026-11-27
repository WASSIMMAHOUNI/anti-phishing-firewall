# Agent 2: Semantic Text Classification (distilbert-base-uncased)

Contextual email subject + body classification agent for PHISHGUARD.

## 1. Model Weights & Placement (`models/distilbert_phishing_model/`)
- Size: `235.7 MB` zipped (`268 MB` unpacked `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `training_args.bin`).
- Download & extract from Release `v1.0-agent2-model`:
```bash
mkdir -p models/distilbert_phishing_model
curl -L -o distilbert_phishing_model.zip [https://github.com/WASSIMMAHOUNI/anti-phishing-firewall/releases/download/v1.0-agent2-model/distilbert_phishing_model.zip](https://github.com/WASSIMMAHOUNI/anti-phishing-firewall/releases/download/v1.0-agent2-model/distilbert_phishing_model.zip)
unzip -oq distilbert_phishing_model.zip -d models/distilbert_phishing_model/
```

## 2. Installation (CPU-only, no GPU required)
```bash
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install -r requirements.txt
```

## 3. Standalone Inference (`text_agent.py`)
- Label mapping: `0 = Safe`, `1 = Not Safe` (softmax index `1`).
- Preprocessing: `f"{subject or ''} {body or ''}".strip()`, WordPiece (`distilbert-base-uncased`), `max_length=256`, truncation=True. Handles `None`, empty, or very long strings.
- Model & tokenizer load ONCE at module level (~89.5 ms/email on AMD Ryzen 5 PRO 4650U CPU).
```python
from text_agent import predict_email
score = predict_email("URGENT: Account Suspended", "Click here to verify your credentials.")
```

## 4. Reference Verification Examples (`python text_agent.py`)
| Case | Subject | Body | Verified `text_score` |
| :--- | :--- | :--- | :---: |
| Empty subject / safe | `""` | `"Team meeting moved to Thursday at 10 AM in Room B2. Please bring your laptop."` | `0.057951` |
| Urgent credential lure | `"URGENT: Account Suspended"` | `"URGENT: Your bank account has been suspended due to unusual login. Click here immediately to verify your credentials."` | `0.999732` |
| Routine project sync | `"Project sync"` | `"Hi team, let us meet tomorrow at 10 AM in room B2."` | `0.007523` |
| Invoice lure | `"Overdue Invoice #4821"` | `"Dear customer, your account will be locked unless you open the attached payment portal and confirm your billing details."` | `0.999779` |
| `email_id=1` (`val`) | *(from `emails_clean.csv`)* | *(from `emails_clean.csv`)* | `0.999836` |
| `email_id=8` (`val`) | *(from `emails_clean.csv`)* | *(from `emails_clean.csv`)* | `0.000171` |
| `email_id=11` (`test`) | *(from `emails_clean.csv`)* | *(from `emails_clean.csv`)* | `0.999729` |

## 5. Offline Predictions (`transformer_scores.csv`) & Agent 3 Schema
- Columns: `email_id,split,text_score` (`14,683` rows: `7,340` `val` + `7,343` `test`; `train` excluded).
- Regenerate without notebook: `python generate_scores.py`
- Agent 3: inner-join on `email_id` with `emails_clean.csv`, train fusion on `split == "val"`, evaluate once on `split == "test"`.

## 6. Saved Evaluation Metrics (`agent2_metrics.json`)
- Validation (`7,340` emails): Accuracy `0.9950`, Precision `0.9945`, Recall `0.9966`, F1 `0.9955`, ROC-AUC `0.9997`, Confusion Matrix: `TN=3181, FP=23, FN=14, TP=4122`.
- Test (`7,343` emails): Held uninspected (`null`) until final Agent 3 evaluation.
