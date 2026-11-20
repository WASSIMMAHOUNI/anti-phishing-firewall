import pandas as pd
from pathlib import Path

EMAIL_DIR = Path("data/emails")

for f in sorted(EMAIL_DIR.glob("*.csv")):
    df = pd.read_csv(f)
    print(f"\n== {f.name}  shape={df.shape}")
    print("label counts:\n", df["label"].value_counts(dropna=False))
    print("nulls:\n", df.isna().sum()[lambda s: s > 0])
    text_col = "text_combined" if "text_combined" in df else "body"
    print("duplicate texts:", df[text_col].duplicated().sum())
    if "urls" in df:
        print("urls value counts:\n", df["urls"].value_counts().head())

web = pd.read_csv(
    "data/webpages/Webpages_Classification_train_data.csv/"
    "Webpages_Classification_train_data.csv",
    nrows=20000,
)
print("\n== webpages (first 20k rows)")
print(web.drop(columns=["content"]).head())
print("label counts:\n", web["label"].value_counts(dropna=False))
print("dtypes:\n", web.dtypes)