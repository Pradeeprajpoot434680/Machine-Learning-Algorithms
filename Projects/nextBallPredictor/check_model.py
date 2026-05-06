import pandas as pd
import numpy as np
import xgboost as xgb

# 1. Load your model
model = xgb.XGBClassifier()
model.load_model("cricket_model.json")

def run_boundary_sweep():
    print("🚀 Starting Model Stress Test...")
    print(f"{'Batter SR':<12} | {'Over':<8} | {'RR':<8} | {'Pred':<6} | {'Prob Distribution (0, 1, 2, 4, 6)'}")
    print("-" * 85)

    # Scenarios to test: We vary the Batter SR from 0.5 to 3.5
    # To see when the model "flips" from predicting 1 to predicting 4/6
    found_boundary = False
    
    for sr in np.arange(0.5, 4.0, 0.3):
        for over_val in [5, 12, 19]: # Powerplay, Middle, Death
            # Constructing the feature vector based on your 12 features:
            # [team, over, ball, batter, bowler, ball_num, score, wicks, rr, w_rem, phase, sr]
            
            ball_num = (over_val * 6) + 1
            curr_score = int(ball_num * 1.5) # Assuming a decent score
            rr = curr_score / (ball_num / 6)
            phase = 0 if over_val < 6 else 1 if over_val < 15 else 2
            
            # Use common IDs (ensure these exist in your LabelEncoder range)
            test_feat = [
                0,          # team
                over_val,   # over
                1,          # ball
                10,         # batter ID
                5,          # bowler ID
                ball_num,   # ball_number
                curr_score, # current_score
                3,          # wickets
                rr,         # run_rate
                7,          # wickets_remaining
                phase,      # phase
                sr          # batter_sr (THE KEY VARIABLE)
            ]

            feat_array = np.array([test_feat])
            prediction = model.predict(feat_array)[0]
            probs = model.predict_proba(feat_array)[0]
            
            # Formatting for display
            prob_str = " | ".join([f"{p:.2f}" for p in probs])
            
            if prediction >= 4:
                found_boundary = True
                prefix = "🔥 MATCH!" 
            else:
                prefix = "  "

            print(f"{sr:<12.1f} | {over_val:<8} | {rr:<8.1f} | {prefix} {prediction:<2} | [{prob_str}]")

    if not found_boundary:
        print("\n❌ RESULT: The model NEVER predicted a boundary.")
        print("REASON: Your model is likely biased towards frequency (0s and 1s are 80% of data).")
        print("FIX: You MUST retrain using 'Sample Weights' to balance the classes.")

if __name__ == "__main__":
    run_boundary_sweep()