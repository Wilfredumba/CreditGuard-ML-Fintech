import pandas as pd
import numpy as np

df = pd.read_csv('data/Loan_Scorecard_DTI.csv')

# 1. Magnification: Let's expand the score spread 
# We'll re-scale the existing scores to use the full 300-850 range
s_min, s_max = df['Credit_Score'].min(), df['Credit_Score'].max()
df['Calibrated_Score'] = 300 + (df['Credit_Score'] - s_min) * (850 - 300) / (s_max - s_min)

# 2. Use Percentiles for Decisions (Industry Standard for Benchmarking)
# Top 70% are Approved, Bottom 10% are Rejected, middle 20% for Review
low_cutoff = df['Calibrated_Score'].quantile(0.10)
high_cutoff = df['Calibrated_Score'].quantile(0.30)

def final_decision(score):
    if score > high_cutoff: return 'Approved'
    elif score > low_cutoff: return 'Manual Review'
    else: return 'Rejected'

df['Decision'] = df['Calibrated_Score'].apply(final_decision)

print(f"--- Calibrated Report ---")
print(f"New Score Range: {df['Calibrated_Score'].min():.0f} to {df['Calibrated_Score'].max():.0f}")
print(f"Approval Threshold: > {high_cutoff:.0f}")
print("\n--- New Portfolio Impact ---")
print(df['Decision'].value_counts(normalize=True) * 100)

df.to_csv('data/CreditGuard_Calibrated.csv', index=False)
