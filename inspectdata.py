import pandas as pd

df = pd.read_csv("dataset.csv")

columns_to_check = [
    "Has_Hypertension",
    "Family_History",
    "Smoking_Status",
    "BP_History"
]

for col in columns_to_check:
    print(f"\n========== {col} ==========")
    print(df[col].value_counts(dropna=False))

print("\n========== FIRST FIVE RECORDS ==========")
print(df.head().to_string(index=False))
