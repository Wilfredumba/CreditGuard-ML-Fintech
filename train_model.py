import numpy as np
import pandas as pd

# 1. Load data
df = pd.read_csv('data/Loan_Default_Clean.csv')

# Select only numerical features for now to keep it simple
features = ['income', 'loan_amount', 'property_value', 'term']
X = df[features].values
y = df['Status'].values.reshape(-1, 1)

# 2. Manual Scaling (Crucial Step!)
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_scaled = (X - X_min) / (X_max - X_min)

# 3. Stable Sigmoid Function
def sigmoid(z):
    # Clips z to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# 4. Training with a smaller Learning Rate
def train(X, y, iterations=2000, lr=0.001):
    weights = np.zeros((X.shape[1], 1))
    n_samples = X.shape[0]
    
    for i in range(iterations):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        
        # Gradient Descent
        gradient = np.dot(X.T, (predictions - y)) / n_samples
        weights -= lr * gradient
        
        if i % 500 == 0:
            loss = -np.mean(y * np.log(predictions + 1e-9) + (1-y) * np.log(1-predictions + 1e-9))
            print(f"Iteration {i}: Loss = {loss:.4f}")
            
    return weights

# 5. Run it
weights = train(X_scaled, y)
print("\nSuccess! Stable Weights (Beta Coefficients):")
for name, w in zip(features, weights):
    print(f"{name}: {w[0]:.4f}")
