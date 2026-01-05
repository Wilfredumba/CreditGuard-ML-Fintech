import numpy as np
import pandas as pd

# 1. Load the cleaned data
df = pd.read_csv('data/Loan_Default_Clean.csv')
features = ['income', 'loan_amount', 'property_value', 'term']
X_raw = df[features].values
y = df['Status'].values.reshape(-1, 1)

# 2. Scale Data (Min-Max)
X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
X_scaled = (X_raw - X_min) / (X_max - X_min)

# 3. Add the BIAS Column (A column of ones)
X_with_bias = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

# 4. Your Best Trained Weights (From your previous successful run)
# Format: [Intercept, income, loan_amount, property_value, term]
weights = np.array([[-0.8737], [-0.0534], [-0.1639], [-0.1267], [-0.2487]])

# --- START SCORECARD PARAMETERS ---

# 5. Define Scorecard Parameters
base_score = 600
pdo = 50  # Points to Double the Odds
factor = pdo / np.log(2)
offset = base_score - (factor * np.log(1)) # Assuming 1:1 odds at base_score

# 6. Calculate the 'Log-Odds' for each person
log_odds = np.dot(X_with_bias, weights)

# 7. Transform Log-Odds to a 300-850 Score
# We use subtraction because higher log-odds = higher risk = lower score
user_scores = offset - (factor * log_odds)

# 8. Clip scores to stay within 300-850 range
user_scores = np.clip(user_scores, 300, 850)

# 9. Show Results
df['Credit_Score'] = user_scores.flatten().astype(int)
print("\n--- Sample Credit Scores ---")
# Showing Status (0=Paid, 1=Default) vs the new Score
print(df[['income', 'loan_amount', 'Status', 'Credit_Score']].head(10))

# 10. Save the final scorecard
df.to_csv('data/Loan_Scorecard_Final.csv', index=False)
print("\nSuccess! Final scorecard saved to data/Loan_Scorecard_Final.csv")
