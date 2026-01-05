import numpy as np

# 1. Model Parameters (From your D1 results)
weights = np.array([[-1.0690], [-0.2708], [-0.6213], [1.1741], [-0.6591], [0.0227]])
# Training Min/Max bounds for scaling (Approx based on your data)
X_min = np.array([0, 0, 0, 0, 0]) 
X_max = np.array([20000, 1000000, 50, 1500000, 360]) # income, loan, dti, prop_val, term

def get_score(income, loan, prop_val, term):
    # Feature Engineering
    dti = loan / (income + 1)
    raw_input = np.array([income, loan, dti, prop_val, term])
    
    # Scale & Add Bias
    scaled = (raw_input - X_min) / (X_max - X_min)
    final_input = np.insert(scaled, 0, 1) # Add Intercept
    
    # Logit -> Raw Score -> Calibration
    log_odds = np.dot(final_input, weights)
    # Using a simple linear mapping based on your previous 300-850 calibration
    # Adjusting score based on your engine's sensitivity
    score = 680 - (log_odds[0] * 50) 
    calibrated = 300 + (score - 660) * (850 - 300) / (700 - 660)
    
    return np.clip(calibrated, 300, 850)

# 3. Interactive Interface
print("--- CreditGuard-ML Live Inference ---")
inc = float(input("Enter Monthly Income: "))
l_amt = float(input("Enter Loan Amount: "))
p_val = float(input("Enter Property Value: "))
trm = float(input("Enter Term (months): "))

final_score = get_score(inc, l_amt, p_val, trm)

print(f"\nResulting Credit Score: {final_score:.0f}")
if final_score > 534:
    print("Decision: APPROVED ✅")
else:
    print("Decision: REJECTED ❌")
