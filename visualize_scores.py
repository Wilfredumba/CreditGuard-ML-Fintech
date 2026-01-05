import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load final results
df = pd.read_csv('data/Loan_Scorecard_Final.csv')

plt.figure(figsize=(10, 6))
sns.boxplot(x='Status', y='Credit_Score', data=df, palette='viridis')
plt.title('Credit Score Distribution: Paid (0) vs Default (1)')
plt.xlabel('Loan Status (0=Paid, 1=Default)')
plt.ylabel('Calculated Credit Score')

# Save the plot to a file
plt.savefig('credit_score_audit.png')
print("Graph saved as credit_score_audit.png. Transfer it to your phone storage to view!")
