import numpy as np
import time

# --- Configuration (Based on your successful Calibration) ---
WEIGHTS = np.array([-1.0690, -0.2708, -0.6213, 1.1741, -0.6591, 0.0227])
X_MIN = np.array([0, 0, 0, 0, 0]) 
X_MAX = np.array([20000, 1000000, 50, 1500000, 360])
CUTOFF = 534

def calculate_score(income, loan, prop_val, term):
    dti = loan / (income + 1)
    raw = np.array([income, loan, dti, prop_val, term])
    scaled = (raw - X_MIN) / (X_MAX - X_MIN)
    final_input = np.insert(scaled, 0, 1) 
    
    log_odds = np.dot(final_input, WEIGHTS)
    # Mapping to your 300-850 scale
    score = 680 - (log_odds * 50) 
    calibrated = 300 + (score - 660) * (850 - 300) / (700 - 660)
    return np.clip(calibrated, 300, 850)

def main():
    print("\033[94m" + "="*40)
    print("   CREDITGUARD ML: DECISION PORTAL")
    print("="*40 + "\033[0m")
    
    try:
        inc = float(input("Applicant Monthly Income (USD): "))
        loan = float(input("Requested Loan Amount: "))
        p_val = float(input("Collateral/Property Value: "))
        trm = float(input("Loan Term (Months): "))
        
        print("\n\033[93mAnalyzing risk profiles...\033[0m")
        time.sleep(1) # Simulates processing
        
        score = calculate_score(inc, loan, p_val, trm)
        
        print("-" * 40)
        print(f"FINAL CREDIT SCORE: \033[1m{int(score)}\033[0m")
        
        if score > 650:
            print("STATUS: \033[92mSTRONGLY APPROVED\033[0m ✅")
        elif score > CUTOFF:
            print("STATUS: \033[96mCONDITIONALLY APPROVED\033[0m ⚠️")
        else:
            print("STATUS: \033[91mREJECTED (High DTI Risk)\033[0m ❌")
        print("-" * 40)

    except ValueError:
        print("\033[91mError: Please enter numerical values only.\033[0m")

if __name__ == "__main__":
    main()
