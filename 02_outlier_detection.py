import numpy as np
import pandas as pd

df = pd.read_excel('loan.xlsx')
print("Columns:", df.columns)


num_cols = df.select_dtypes(include=np.number).columns.drop(target_col)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} outliers")
