import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="T20 Ball Predictor", 
    page_icon="", 
    layout="wide"
)

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    """
    Loads model, data, and fits encoders to ensure 
    consistent mapping between names and numbers.
    """
    try:
        # 1. Load your trained XGBoost model
        model = xgb.XGBClassifier()
        model.load_model("cricket_model.json") 
        
        # 2. Load your dataset for metadata and encoding
        df = pd.read_parquet("cricket_dataset.parquet")
        
        # 3. Create lookup dictionaries for player stats
        # These are used to feed the 'batter_avg' feature
        batter_stats = df.groupby('batter')['runs'].mean().to_dict()
        
        # 4. Initialize and Fit Encoders
        # This converts names like "Virat Kohli" into the exact integer IDs
        # the model was trained on.
        le_team = LabelEncoder().fit(df['team'].astype(str))
        le_batter = LabelEncoder().fit(df['batter'].astype(str))
        le_bowler = LabelEncoder().fit(df['bowler'].astype(str))
        
        # Get unique run outcomes for the chart (usually 0, 1, 2, 3, 4, 6)
        outcomes = sorted(df['runs'].unique())
        
        return model, batter_stats, le_team, le_batter, le_bowler, outcomes, df
    
    except FileNotFoundError:
        st.error("Missing files! Ensure 'cricket_model.json' and 'cricket_dataset.parquet' are in the same folder.")
        st.stop()

# Initialize resources
model, batter_stats, le_team, le_batter, le_bowler, outcomes, raw_df = load_resources()

# --- UI HEADER ---
st.title("T20 Next Ball Run Predictor")
st.markdown("This AI model analyzes match context and player history to predict the outcome of the next delivery.")
st.markdown("---")

# --- INPUT SECTION ---
# Sidebar for Team Selection
st.sidebar.header("Team Selection")
bat_team_name = st.sidebar.selectbox("Batting Team", sorted(le_team.classes_))
bowl_team_name = st.sidebar.selectbox("Bowling Team", sorted([t for t in le_team.classes_ if t != bat_team_name]))

# Main columns for match state and matchups
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Match State")
    current_score = st.number_input("Current Score", min_value=0, value=100, step=1)
    wickets = st.number_input("Wickets Fallen", 0, 9, 3)
    over = st.slider("Over Number (0-19)", 0, 19, 10)
    ball = st.slider("Ball in Over (1-6)", 1, 6, 1)

with col2:
    st.subheader("Player Matchup")
    striker_name = st.selectbox("Select Striker (Batter)", sorted(le_batter.classes_))
    bowler_name = st.selectbox("Select Bowler", sorted(le_bowler.classes_))

with col3:
    st.subheader(" Prediction")
    st.write("Calculate probabilities for the next delivery.")
    submit = st.button("Predict Outcome", use_container_width=True, type="primary")

# --- PREDICTION LOGIC ---
if submit:
    # 1. Feature Engineering (Derived Features)
    ball_number = (over * 6) + ball
    run_rate = current_score / (ball_number / 6 + 0.1)
    wickets_remaining = 10 - wickets
    
    # Calculate Phase (Powerplay, Middle, Death)
    if over <= 5: phase = 0 
    elif over <= 14: phase = 1 
    else: phase = 2 

    # Lookup Batter Skill (Mean runs per ball)
    # Default to global average if player not found
    b_avg = batter_stats.get(striker_name, raw_df['runs'].mean())
    
    # 2. String-to-Integer Encoding
    try:
        team_enc = le_team.transform([bat_team_name])[0]
        batter_enc = le_batter.transform([striker_name])[0]
        bowler_enc = le_bowler.transform([bowler_name])[0]
    except ValueError:
        st.error("Encoding Error: Selection not recognized by the model.")
        st.stop()

    # 3. Construct Final Feature Array
    # MUST match the 12-feature order used during model training
    feature_list = [
        team_enc, over, ball, batter_enc, bowler_enc, ball_number,
        current_score, wickets, run_rate, wickets_remaining, phase, b_avg
    ]
    
    features_array = np.array([feature_list])

    # 4. Model Inference
    prediction = model.predict(features_array)[0]
    probs = model.predict_proba(features_array)[0]
    
    # --- RESULT DISPLAY ---
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.write("### Prediction")
        st.metric(label="Most Likely Outcome", value=f"{int(prediction)} Runs")
        
        # Risk & Context Analysis
        boundary_prob = (probs[4] + (probs[6] if 6 in outcomes else 0)) * 100
        dot_prob = probs[0] * 100
        
        st.write(f"**Boundary Chance:** `{boundary_prob:.1f}%`")
        st.write(f"**Dot Ball Chance:** `{dot_prob:.1f}%`")
        
        if boundary_prob > 25:
            st.warning("Aggressive context detected! High boundary probability.")
        elif dot_prob > 45:
            st.info("🛡️ Defensive context detected. High dot ball probability.")

    with res_col2:
        st.write("### Outcome Probabilities")
        # Map outcomes to probabilities for the chart
        prob_df = pd.DataFrame({
            'Outcome (Runs)': [str(o) for o in outcomes],
            'Probability (%)': [p * 100 for p in probs]
        })
        st.bar_chart(prob_df.set_index('Outcome (Runs)'))