import pandas as pd

# 1. Define the exact file name from your folder
file_name = "phishing_legit_dataset_KD_10000.csv"

# 2. Load the dataset into a pandas DataFrame
df = pd.read_csv(file_name)

# 3. Print basic information
print("--- DATASET OVERVIEW ---")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

# 4. Print the exact column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 5. Show the first 3 rows to see the actual text and labels
print("\n--- FIRST 3 ROWS ---")
print(df.head(3))

# 6. Check the distribution of the main label column
# We will guess the column is named 'label' or 'Label' or 'Phishing_Type'
print("\n--- LABEL DISTRIBUTION ---")
for col in ['label', 'Label', 'target', 'Target', 'phishing_type']:
    if col in df.columns:
        print(f"Counts for '{col}':")
        print(df[col].value_counts(dropna=False))
        print("-" * 20)