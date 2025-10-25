import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_excel('loan.xlsx')

# 1. Audit the dataset
print("Columns:", df.columns.tolist())
print("Data Types:\n", df.dtypes)
print("Shape (rows, columns):", df.shape)

# Show head for quick preview
print("Preview:\n", df.head())

# Check for missing values
missing_count = df.isnull().sum()
print("Missing Values per Column:\n", missing_count)

# Optionally, view % missing per column
print("Missing % per Column:\n", missing_count / len(df) * 100)

# Check for duplicates
dup_count = df.duplicated().sum()
print("Number of duplicate rows:", dup_count)

# Remove duplicate rows
df_cleaned = df.drop_duplicates()
print("Shape after duplicate removal:", df_cleaned.shape)

# 2.1 - Handle missing values (impute, drop, infer)
# For example: fill numeric missing with median, categorical with mode
for col in df_cleaned.columns:
    if df_cleaned[col].isnull().sum() > 0:
        if df_cleaned[col].dtype in [np.float64, np.int64]:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        else:
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

# 2.2 - Identify outliers (using IQR for numeric features)
num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
for col in num_cols:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_cleaned[(df_cleaned[col] < Q1 - 1.5 * IQR) | (df_cleaned[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(outliers)} outliers found.")

    # Optional: cap outliers to boundary values
    df_cleaned[col] = np.where(df_cleaned[col] < Q1 - 1.5 * IQR, Q1 - 1.5 * IQR,
                          np.where(df_cleaned[col] > Q3 + 1.5 * IQR, Q3 + 1.5 * IQR, df_cleaned[col]))

# 2.3 - Standardize formatting (for categories; example for Gender)
df_cleaned['Gender'] = df_cleaned['Gender'].str.strip().str.upper().replace({'FEMALE':'F', 'MALE':'M'})
print("Unique values for Gender after standardization:", df_cleaned['Gender'].unique())

# Save cleaned dataframe (optional)
df_cleaned.to_excel('loan_cleaned.xlsx', index=False)
