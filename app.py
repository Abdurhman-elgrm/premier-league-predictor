"""
Premier League Match Predictor - Interactive Streamlit Web App
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from predict import predict_match

st.set_page_config(
    page_title="Premier League Match & Score Predictor",
    page_icon="⚽",
    layout="centered"
)

# Header
st.title("⚽ Premier League Match & Score Predictor")
st.markdown("Predict **Exact Scores**, **Expected Goals (xG)**, and **Win/Draw/Loss Probabilities** for any Premier League fixture.")

# Exact 20-Team League Roster
PL_TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Coventry",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Hull",
    "Ipswich",
    "Leeds",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle",
    "Nott'm Forest",
    "Sunderland",
    "Tottenham Hotspur"
]

st.write("---")

# Selection columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_idx = PL_TEAMS.index("Arsenal")
    home_team = st.selectbox("Select Home Club", PL_TEAMS, index=home_idx, key="home_select")

with col2:
    st.subheader("✈️ Away Team")
    away_idx = PL_TEAMS.index("Chelsea")
    away_team = st.selectbox("Select Away Club", PL_TEAMS, index=away_idx, key="away_select")

st.write("")

if st.button("🔮 Predict Match Score & Outcome", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("⚠️ Please choose two different teams for the match!")
    else:
        with st.spinner(f"Analyzing {home_team} vs {away_team}..."):
            try:
                res = predict_match(home_team, away_team)
                
                st.write("---")
                
                # 1. SCOREBOARD BANNER
                st.markdown(
                    f"""
                    <div style="background-color: #1e1e2f; padding: 25px; border-radius: 12px; text-align: center; color: white; margin-bottom: 20px;">
                        <h4 style="margin: 0; color: #a0a0b0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Predicted Full-Time Score</h4>
                        <div style="display: flex; justify-content: space-around; align-items: center; margin-top: 15px;">
                            <div style="flex: 1;">
                                <h2 style="margin: 0; font-size: 24px; color: #ffffff;">{home_team}</h2>
                                <p style="margin: 5px 0 0 0; color: #4CAF50; font-size: 14px; font-weight: bold;">xG: {res['exp_home_goals']:.2f}</p>
                            </div>
                            <div style="background: #e91e63; padding: 10px 25px; border-radius: 8px; font-size: 36px; font-weight: bold; letter-spacing: 4px;">
                                {res['predicted_score']}
                            </div>
                            <div style="flex: 1;">
                                <h2 style="margin: 0; font-size: 24px; color: #ffffff;">{away_team}</h2>
                                <p style="margin: 5px 0 0 0; color: #4CAF50; font-size: 14px; font-weight: bold;">xG: {res['exp_away_goals']:.2f}</p>
                            </div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # 2. PROBABILITY METRICS
                st.subheader("📊 Match Outcome Probabilities")
                probs = res['probabilities']
                m1, m2, m3 = st.columns(3)
                m1.metric(label=f"🏠 {home_team} Win", value=f"{probs['Home Win']*100:.1f}%")
                m2.metric(label="🤝 Draw", value=f"{probs['Draw']*100:.1f}%")
                m3.metric(label=f"✈️ {away_team} Win", value=f"{probs['Away Win']*100:.1f}%")
                
                # Visual Bar Chart
                prob_df = pd.DataFrame({
                    "Outcome": [f"Home ({home_team})", "Draw", f"Away ({away_team})"],
                    "Probability (%)": [probs['Home Win']*100, probs['Draw']*100, probs['Away Win']*100]
                })
                st.bar_chart(prob_df.set_index("Outcome"), color="#1f77b4")
                
                # 3. TOP SCORE MATRIX
                st.subheader("🎯 Top Most Likely Scorelines")
                score_cols = st.columns(len(res['top_scores'][:4]))
                for i, (score, prob) in enumerate(res['top_scores'][:4]):
                    with score_cols[i]:
                        st.metric(label=f"#{i+1} Scoreline", value=score, delta=f"{prob*100:.1f}%")

            except Exception as e:
                st.error(f"Prediction error: {e}")

st.write("---")
st.caption("Premier League Predictor • Trained on 25 seasons of historical match data (2000–2026)")
