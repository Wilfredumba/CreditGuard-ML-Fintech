import numpy as np
import pandas as pd

# Load and Prep
df = pd.read_csv('data/Loan_Default_Clean.csv')
features = ['income', 'loan_amount', 'property_value', 'term']
X = df[features].values
y = df['Status'].values.reshape(-1, 1)

# 1. Scale Data (Min-Max)
X_min, X_max = X.min(axis=0), X.max(axis=0)
X_scaled = (X - X_min) / (X_max - X_min)

# 2. Add the BIAS Column (A column of ones)
# This represents beta_0 in: y = beta_0 + beta_1*X1 + ...
X_with_bias = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def train(X, y, iterations=3000, lr=0.05): # Increased LR slightly
    weights = np.zeros((X.shape[1], 1))
    n_samples = X.shape[0]
    
    for i in range(iterations):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        gradient = np.dot(X.T, (predictions - y)) / n_samples
        weights -= lr * gradient
        
        if i % 500 == 0:
            loss = -np.mean(y * np.log(predictions + 1e-9) + (1-y) * np.log(1-predictions + 1e-9))
            print(f"Iteration {i}: Loss = {loss:.4f}")
            
    return weights

# 3. Run and Interpret
weights = train(X_with_bias, y)

print("\n--- Final Econometric Scorecard ---")
print(f"Intercept (Baseline Risk): {weights[0][0]:.4f}")
for name, w in zip(features, weights[1:]):
    print(f"{name}: {w[0]:.4f}")
