import pandas as pd
import numpy as np

# Load the raw data
df = pd.read_csv('data/Loan_Default.csv')

# 1. Handle Categorical Data (Mode Imputation)
categorical_cols = ['loan_limit', 'approv_in_adv', 'loan_purpose', 'age']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 2. Handle Numerical Data (Median Imputation)
# Median is better than Mean because interest rates and income are usually skewed
numerical_cols = ['Upfront_charges', 'Interest_rate_spread', 'rate_of_interest', 
                  'dtir1', 'income', 'property_value', 'LTV']
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# 3. Handle 'term' (Use a constant like 360 months if missing)
df['term'] = df['term'].fillna(360)

# 4. Final check - Drop any remaining tiny rows (e.g., submission_of_application)
df.dropna(inplace=True)

# 5. Save the Cleaned Version
df.to_csv('data/Loan_Default_Clean.csv', index=False)
print("Success! Cleaned data saved as: data/Loan_Default_Clean.csv")
print(f"Final dataset size: {df.shape}")
