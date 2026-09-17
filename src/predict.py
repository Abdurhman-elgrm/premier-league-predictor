import joblib
import pandas as pd
import numpy as np
import math
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model_v1.pkl"
SCORE_MODEL_PATH = BASE_DIR / "models" / "score_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "matches_features.csv"

# Feature columns expected by the models
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


def poisson_pmf(k: int, lam: float) -> float:
    """Calculates Poisson probability for k goals with expected mean lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)


def get_latest_team_stats(team_name: str, df: pd.DataFrame) -> dict:
    """Fetches the most recent pre-match stats for a team."""
    team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].sort_values('Date')
    if team_matches.empty:
        available = sorted(list(set(df['HomeTeam']).union(set(df['AwayTeam']))))
        raise ValueError(f"Team '{team_name}' not found. Available teams: {', '.join(available[:10])}...")
    
    last_match = team_matches.iloc[-1]
    is_home = (last_match['HomeTeam'] == team_name)
    
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


def predict_match(home_team: str, away_team: str) -> dict:
    # 1. Load models and dataset
    outcome_model = joblib.load(MODEL_PATH)
    score_models = joblib.load(SCORE_MODEL_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # 2. Get latest stats
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
    
    # 5. Outcome Probabilities: [Home Win, Draw, Away Win]
    probs = outcome_model.predict_proba(X_input)[0]
    
    # 6. Expected Goals (xG)
    exp_h_goals = float(score_models["home_goals_model"].predict(X_input)[0])
    exp_a_goals = float(score_models["away_goals_model"].predict(X_input)[0])
    
    # 7. Exact Score Probabilities (0-0 up to 5-5)
    score_grid = {}
    for h in range(6):
        for a in range(6):
            p = poisson_pmf(h, exp_h_goals) * poisson_pmf(a, exp_a_goals)
            score_grid[f"{h} - {a}"] = float(p)
            
    # Normalize score grid
    total_grid_prob = sum(score_grid.values())
    for k in score_grid:
        score_grid[k] /= total_grid_prob
        
    most_likely_score = max(score_grid, key=score_grid.get)
    top_scores = sorted(score_grid.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Print clean summary
    print(f"\n=======================================================")
    print(f"      PREDICTION: {home_team} vs {away_team}")
    print(f"=======================================================")
    print(f"  PREDICTED SCORE:     {home_team} {most_likely_score} {away_team}")
    print(f"  Expected Goals (xG): {home_team} {exp_h_goals:.2f} - {exp_a_goals:.2f} {away_team}")
    print(f"-------------------------------------------------------")
    print(f"  Outcome Probabilities:")
    print(f"    Home Win ({home_team}): {probs[0]*100:.1f}%")
    print(f"    Draw:                  {probs[1]*100:.1f}%")
    print(f"    Away Win ({away_team}): {probs[2]*100:.1f}%")
    print(f"-------------------------------------------------------")
    print(f"  Top Most Likely Scorelines:")
    for score, prob in top_scores[:3]:
        print(f"    {score:<7} -> {prob*100:.1f}%")
    print(f"=======================================================\n")
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_score": most_likely_score,
        "exp_home_goals": round(exp_h_goals, 2),
        "exp_away_goals": round(exp_a_goals, 2),
        "probabilities": {
            "Home Win": float(probs[0]),
            "Draw": float(probs[1]),
            "Away Win": float(probs[2])
        },
        "top_scores": top_scores
    }


if __name__ == "__main__":
    predict_match("Arsenal", "Chelsea")
