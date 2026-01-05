import pandas as pd

# Load data
df = pd.read_csv('data/Loan_Default.csv')

# 1. Total Missing Values per column
null_counts = df.isnull().sum()
null_percent = (null_counts / len(df)) * 100

# 2. Combine into a Summary Table
audit_report = pd.DataFrame({
    'Missing Values': null_counts,
    'Percentage (%)': null_percent
}).sort_values(by='Percentage (%)', ascending=False)

# 3. Print only columns that have missing data
print("--- Data Audit: Missing Values Summary ---")
print(audit_report[audit_report['Missing Values'] > 0])

# 4. Check for duplicates (common in financial data)
print(f"\nDuplicate Rows Found: {df.duplicated().sum()}")
