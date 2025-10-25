import pandas as pd

df = pd.read_excel('loan.xlsx')
print("Columns:", df.columns)
target_col = None
for col in df.columns:
    if col.lower().replace('_', '').replace(' ', '') in ['loanstatus']:
        target_col = col
        break
if target_col is None:
    raise Exception("Target column 'Loan_Status' not found!")

print("Target column:", target_col)
from sklearn.preprocessing import LabelEncoder

if df[target_col].dtype == 'object':
    df[target_col] = df[target_col].str.strip()
    df[target_col] = LabelEncoder().fit_transform(df[target_col])
print("Encoded target unique values:", df[target_col].unique())
for col in df.columns:
    if col != target_col and df[col].dtype == 'object':
        vals = df[col].dropna().unique()
        # Binary Yes/No
        if sorted([str(v).strip() for v in vals]) == ['No', 'Yes']:
            df[col] = df[col].map({'No': 0, 'Yes': 1})
        elif len(vals) <= 10:  # Small set, label encode
            df[col] = LabelEncoder().fit_transform(df[col])
        else:  # High-cardinality object col: drop it
            print(f"Dropped high-cardinality column: {col}")
            df.drop(col, axis=1, inplace=True)
print("Columns after binary encoding:", df.columns.tolist())
