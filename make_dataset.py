import pandas as pd
from pathlib import Path

SEED = 42  # fixed number so the split is the same every time we run it
SOURCES = ["CEAS_08", "Nazario", "Nigerian_Fraud", "SpamAssasin"]
IN_DIR = Path("data/emails")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Load the 4 files and stack them into one table
frames = []
for name in SOURCES:
    df = pd.read_csv(IN_DIR / f"{name}.csv")
    df["source"] = name          # remember where each email came from
    frames.append(df)
emails = pd.concat(frames, ignore_index=True)
print("loaded:", len(emails))

# 2) Drop emails with no body text (nothing for the Transformer to read)
emails["body"] = emails["body"].fillna("")
emails = emails[emails["body"].str.strip() != ""]
print("after dropping empty bodies:", len(emails))

# 3) Make a "fingerprint" of each body to find duplicates
emails["fingerprint"] = (
    emails["body"].str.lower().str.split().str.join(" ")
)

# 3a) Same body but different labels = confusing, remove all of them
labels_per_body = emails.groupby("fingerprint")["label"].nunique()
conflicting = labels_per_body[labels_per_body > 1].index
emails = emails[~emails["fingerprint"].isin(conflicting)]
print("after removing conflicting duplicates:", len(emails))

# 3b) Same body and same label = keep just one copy
emails = emails.drop_duplicates(subset="fingerprint", keep="first")
print("after removing duplicates:", len(emails))

# 4) Give each email a unique id
emails = emails.reset_index(drop=True)
emails["email_id"] = emails.index

# 5) Assign the split ONCE: 70% train, 15% val, 15% test
#    Done separately for label 0 and label 1 so each split keeps similar balance
emails["split"] = ""
for label, group in emails.groupby("label"):
    shuffled = group.sample(frac=1, random_state=SEED)
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    emails.loc[shuffled.index[:n_train], "split"] = "train"
    emails.loc[shuffled.index[n_train:n_train + n_val], "split"] = "val"
    emails.loc[shuffled.index[n_train + n_val:], "split"] = "test"

# 6) Save (fingerprint was only a helper, so we drop it)
cols = ["email_id", "split", "source", "sender", "receiver",
        "date", "subject", "body", "urls", "label"]
emails[cols].to_csv(OUT_DIR / "emails_clean.csv", index=False)

print("\nsaved to", OUT_DIR / "emails_clean.csv")
print("\nrows per split and label:")
print(emails.groupby(["split", "label"]).size())
print("\nrows per source and split:")
print(emails.groupby(["source", "split"]).size())