
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model_v1.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "matches_features.csv"

# Feature columns expected by the model
FEATURE_COLS = [
    'HomeElo', 'AwayElo', 'EloDiff',
    'Home_Form_5', 'Away_Form_5', 'Diff_Form_5',
    'Home_GoalsFor_5', 'Away_GoalsFor_5', 'Diff_GoalsFor_5',
    'Home_GoalsAgainst_5', 'Away_GoalsAgainst_5', 'Diff_GoalsAgainst_5',
    'Home_SoT_For_5', 'Away_SoT_For_5', 'Diff_SoT_For_5',
    'Home_Rest_Days', 'Away_Rest_Days', 'Diff_Rest_Days',
    'Home_Cum_Points', 'Away_Cum_Points', 'Diff_Cum_Points',
    'Home_Table_Pos', 'Away_Table_Pos', 'Diff_Table_Pos',
    'Home_venue_effect', 'Away_venue_effect', 'h2h_home_winrate'
]


def get_latest_team_stats(team_name, df):
    """Fetches the most recent pre-match stats for a team."""
    # Look for the team's most recent match as either Home or Away
    team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].sort_values('Date')
    if team_matches.empty:
        raise ValueError(f"Team '{team_name}' not found in dataset!")
    
    last_match = team_matches.iloc[-1]
    is_home = last_match['HomeTeam'] == team_name
    
    return {
        'Elo': last_match['HomeElo'] if is_home else last_match['AwayElo'],
        'Form_5': last_match['Home_Form_5'] if is_home else last_match['Away_Form_5'],
        'GoalsFor_5': last_match['Home_GoalsFor_5'] if is_home else last_match['Away_GoalsFor_5'],
        'GoalsAgainst_5': last_match['Home_GoalsAgainst_5'] if is_home else last_match['Away_GoalsAgainst_5'],
        'SoT_For_5': last_match['Home_SoT_For_5'] if is_home else last_match['Away_SoT_For_5'],
        'Rest_Days': 7.0,
        'Cum_Points': last_match['Home_Cum_Points'] if is_home else last_match['Away_Cum_Points'],
        'Table_Pos': last_match['Home_Table_Pos'] if is_home else last_match['Away_Table_Pos'],
        'Venue_Effect': last_match['Home_venue_effect'] if is_home else last_match['Away_venue_effect']
    }


def predict_match(home_team: str, away_team: str):
    # 1. Load model and dataset
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # 2. Get latest stats for both teams
    home_stats = get_latest_team_stats(home_team, df)
    away_stats = get_latest_team_stats(away_team, df)
    
    # 3. Head to Head key
    pair = sorted([home_team, away_team])
    h2h_key = f"{pair[0]}_vs_{pair[1]}"
    h2h_matches = df[df['H2H_key'] == h2h_key]
    h2h_winrate = h2h_matches['h2h_home_winrate'].iloc[-1] if not h2h_matches.empty else 0.45
    
    # 4. Build feature vector
    input_dict = {
        'HomeElo': home_stats['Elo'],
        'AwayElo': away_stats['Elo'],
        'EloDiff': home_stats['Elo'] - away_stats['Elo'],
        'Home_Form_5': home_stats['Form_5'],
        'Away_Form_5': away_stats['Form_5'],
        'Diff_Form_5': home_stats['Form_5'] - away_stats['Form_5'],
        'Home_GoalsFor_5': home_stats['GoalsFor_5'],
        'Away_GoalsFor_5': away_stats['GoalsFor_5'],
        'Diff_GoalsFor_5': home_stats['GoalsFor_5'] - away_stats['GoalsFor_5'],
        'Home_GoalsAgainst_5': home_stats['GoalsAgainst_5'],
        'Away_GoalsAgainst_5': away_stats['GoalsAgainst_5'],
        'Diff_GoalsAgainst_5': home_stats['GoalsAgainst_5'] - away_stats['GoalsAgainst_5'],
        'Home_SoT_For_5': home_stats['SoT_For_5'],
        'Away_SoT_For_5': away_stats['SoT_For_5'],
        'Diff_SoT_For_5': home_stats['SoT_For_5'] - away_stats['SoT_For_5'],
        'Home_Rest_Days': home_stats['Rest_Days'],
        'Away_Rest_Days': away_stats['Rest_Days'],
        'Diff_Rest_Days': 0.0,
        'Home_Cum_Points': home_stats['Cum_Points'],
        'Away_Cum_Points': away_stats['Cum_Points'],
        'Diff_Cum_Points': home_stats['Cum_Points'] - away_stats['Cum_Points'],
        'Home_Table_Pos': home_stats['Table_Pos'],
        'Away_Table_Pos': away_stats['Table_Pos'],
        'Diff_Table_Pos': away_stats['Table_Pos'] - home_stats['Table_Pos'],
        'Home_venue_effect': home_stats['Venue_Effect'],
        'Away_venue_effect': away_stats['Venue_Effect'],
        'h2h_home_winrate': h2h_winrate
    }
    
    X_input = pd.DataFrame([input_dict])[FEATURE_COLS]
    
    # 5. Predict probabilities: [Class 0: Home Win, Class 1: Draw, Class 2: Away Win]
    probs = model.predict_proba(X_input)[0]
    
    print(f"\n==========================================")
    print(f"      PREDICTION: {home_team} vs {away_team}")
    print(f"==========================================")
    print(f"  ⚽ Home Win ({home_team}): {probs[0]*100:.1f}%")
    print(f"  🤝 Draw:                  {probs[1]*100:.1f}%")
    print(f"  ⚽ Away Win ({away_team}): {probs[2]*100:.1f}%")
    print(f"==========================================\n")
    
    return {"Home Win": probs[0], "Draw": probs[1], "Away Win": probs[2]}


if __name__ == "__main__":
    # Test an example match!
    predict_match("Arsenal", "Chelsea")
