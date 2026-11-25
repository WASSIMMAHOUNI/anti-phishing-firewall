import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/processed/emails_clean.csv")

# Combine subject and body into one text
df["text"] = df["subject"].fillna("") + " " + df["body"]

train = df[df["split"] == "train"]
val = df[df["split"] == "val"]

# Turn text into numbers (learn the vocabulary from TRAIN only)
vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3)
X_train = vec.fit_transform(train["text"])
X_val = vec.transform(val["text"])

# Train a simple classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train["label"])

# Check how it does on the validation rows
pred = model.predict(X_val)
print(classification_report(val["label"], pred, digits=3))
print(confusion_matrix(val["label"], pred))

val = val.assign(pred=pred)
for src, g in val.groupby("source"):
    acc = (g["pred"] == g["label"]).mean()
    print(src, len(g), round(acc, 3))

print("\n--- leave-one-source-out test ---")
for held in ["CEAS_08", "SpamAssasin"]:
    tr = df[(df["split"] == "train") & (df["source"] != held)]
    te = df[(df["split"] == "val") & (df["source"] == held)]
    v = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3)
    m = LogisticRegression(max_iter=1000)
    m.fit(v.fit_transform(tr["text"]), tr["label"])
    p = m.predict(v.transform(te["text"]))
    print(held, "held out | accuracy:", round((p == te["label"]).mean(), 3))
    print(confusion_matrix(te["label"], p))