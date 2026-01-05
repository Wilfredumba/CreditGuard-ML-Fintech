import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Feature Engineering
df = pd.read_csv('data/Loan_Default_Clean.csv')

# Create DTI Ratio: Loan Amount / Income (adding 1 to avoid division by zero)
df['DTI_Ratio'] = df['loan_amount'] / (df['income'] + 1)

# Include 'dtir1' if it exists in your dataset as it's a direct debt-to-income metric
features = ['income', 'loan_amount', 'DTI_Ratio', 'property_value', 'term']
X_raw = df[features].values
y = df['Status'].values.reshape(-1, 1)

# 2. Manual Min-Max Scaling
X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
X_scaled = (X_raw - X_min) / (X_max - X_min)
X_with_bias = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

# 3. Training with more iterations to sharpen weights
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def train(X, y, iterations=10000, lr=0.1): # Increased iterations and LR
    weights = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        z = np.dot(X, weights)
        errors = sigmoid(z) - y
        weights -= (lr * np.dot(X.T, errors) / len(y))
    return weights

print("Training model with DTI Ratio...")
weights = train(X_with_bias, y)

# 4. Scorecard Scaling (300-850)
base_score, pdo = 600, 50
factor = pdo / np.log(2)
offset = base_score - (factor * np.log(1))

log_odds = np.dot(X_with_bias, weights)
user_scores = offset - (factor * log_odds)
df['Credit_Score'] = np.clip(user_scores, 300, 850).astype(int)

# 5. Save and Export
df.to_csv('data/Loan_Scorecard_DTI.csv', index=False)
print("--- Weights with DTI ---")
for name, w in zip(['Intercept'] + features, weights):
    print(f"{name}: {w[0]:.4f}")
