import pandas as pd
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "matches_clean.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "matches_features.csv"

print("Loading cleaned match data...")
df = pd.read_csv(INPUT_PATH, dtype={"Season_code": str}, low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 1. Match Points
def points_home(row):
    if row['FTR'] == 'H': return 3
    elif row['FTR'] == 'D': return 1
    return 0

def points_away(row):
    if row['FTR'] == 'A': return 3
    elif row['FTR'] == 'D': return 1
    return 0

df["HomePoints"] = df.apply(points_home, axis=1)
df["AwayPoints"] = df.apply(points_away, axis=1)

# 2. Build Team History
home_records = df[['Season_code', 'Date', 'HomeTeam', 'HomePoints', 'FTHG', 'FTAG', 'HST', 'AST']].copy()
home_records.columns = ['Season_code', 'Date', 'Team', 'Points', 'GoalsFor', 'GoalsAgainst', 'ShotsOnTargetFor', 'ShotsOnTargetAgainst']

away_records = df[['Season_code', 'Date', 'AwayTeam', 'AwayPoints', 'FTAG', 'FTHG', 'AST', 'HST']].copy()
away_records.columns = ['Season_code', 'Date', 'Team', 'Points', 'GoalsFor', 'GoalsAgainst', 'ShotsOnTargetFor', 'ShotsOnTargetAgainst']

team_history = pd.concat([home_records, away_records]).sort_values(['Team', 'Date']).reset_index(drop=True)
team_history['Date'] = pd.to_datetime(team_history['Date'])

# 3. Compute Team-Level Rolling Features
print("Computing rolling team statistics...")

# 5-match rolling form points
team_history['Form_points_last_5'] = (
    team_history.groupby('Team')["Points"]
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum())
)

# 5-match rolling goals scored & conceded
team_history['GoalsFor_last_5'] = (
    team_history.groupby('Team')["GoalsFor"]
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)
team_history['GoalsAgainst_last_5'] = (
    team_history.groupby('Team')["GoalsAgainst"]
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

# 5-match rolling shots on target
team_history['SoT_For_5'] = (
    team_history.groupby('Team')["ShotsOnTargetFor"]
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)
team_history['SoT_Against_5'] = (
    team_history.groupby('Team')["ShotsOnTargetAgainst"]
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

# Rest Days (Fatigue)
team_history['Rest_Days'] = (
    team_history.groupby('Team')['Date'].diff().dt.days
).fillna(7)

# Cumulative Season Points
team_history['Cum_Season_Points'] = (
    team_history.groupby(['Season_code', 'Team'])['Points']
    .transform(lambda x: x.shift(1).cumsum())
    .fillna(0)
)

# 4. Compute Official As-Of-Date Premier League Table Position (1st to 20th)
print("Computing official league table position at game time...")
home_table_pos = []
away_table_pos = []

for season in df['Season_code'].unique():
    season_df = df[df['Season_code'] == season]
    season_teams = sorted(list(set(season_df['HomeTeam']).union(set(season_df['AwayTeam']))))
    
    # Standings table tracking all teams in this season
    standings = {team: {'points': 0, 'gd': 0, 'gf': 0} for team in season_teams}
    
    for idx, row in season_df.iterrows():
        # Rank all teams by Points (descending), Goal Diff (descending), Goals For (descending)
        ranked = sorted(
            season_teams, 
            key=lambda t: (standings[t]['points'], standings[t]['gd'], standings[t]['gf']), 
            reverse=True
        )
        rank_map = {team: rank + 1 for rank, team in enumerate(ranked)}
        
        # Pre-match positions (1 to 20)
        home_pos = rank_map[row['HomeTeam']]
        away_pos = rank_map[row['AwayTeam']]
        home_table_pos.append(home_pos)
        away_table_pos.append(away_pos)
        
        # Update standings after the match
        h_pts = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
        a_pts = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
        h_g = row['FTHG'] if pd.notnull(row['FTHG']) else 0
        a_g = row['FTAG'] if pd.notnull(row['FTAG']) else 0
        
        standings[row['HomeTeam']]['points'] += h_pts
        standings[row['HomeTeam']]['gd'] += (h_g - a_g)
        standings[row['HomeTeam']]['gf'] += h_g
        
        standings[row['AwayTeam']]['points'] += a_pts
        standings[row['AwayTeam']]['gd'] += (a_g - h_g)
        standings[row['AwayTeam']]['gf'] += a_g

df['Home_Table_Pos'] = home_table_pos
df['Away_Table_Pos'] = away_table_pos
df['Diff_Table_Pos'] = df['Away_Table_Pos'] - df['Home_Table_Pos']  # Positive = Home is higher in the table

# 5. Compute Match-Level Features (Elo & Venue)
print("Computing Elo ratings and venue effects...")

unique_teams = set(df['HomeTeam']).union(set(df['AwayTeam']))
elo_ratings = {team: 1500.0 for team in unique_teams}
home_elo = []
away_elo = []
k_factor = 20.0
home_advantage = 65.0

for idx, row in df.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    
    h_rating = elo_ratings[home_team]
    a_rating = elo_ratings[away_team]
    home_elo.append(h_rating)
    away_elo.append(a_rating)
    
    expected_home = 1.0 / (1.0 + 10.0 ** ((a_rating - (h_rating + home_advantage)) / 400.0))
    expected_away = 1.0 - expected_home
    
    if row['FTR'] == 'H':
        s_home, s_away = 1.0, 0.0
    elif row['FTR'] == 'D':
        s_home, s_away = 0.5, 0.5
    else:
        s_home, s_away = 0.0, 1.0
        
    elo_ratings[home_team] = h_rating + k_factor * (s_home - expected_home)
    elo_ratings[away_team] = a_rating + k_factor * (s_away - expected_away)

df['HomeElo'] = home_elo
df['AwayElo'] = away_elo
df['EloDiff'] = df['HomeElo'] - df['AwayElo']

# Venue Effect
df['Home_venue_effect'] = (
    df.groupby('HomeTeam')['HomePoints']
    .transform(lambda x: (x.shift(1) == 3).astype(float).rolling(window=5, min_periods=1).mean())
    .fillna(0.33)
)
df['Away_venue_effect'] = (
    df.groupby('AwayTeam')['AwayPoints']
    .transform(lambda x: (x.shift(1) == 3).astype(float).rolling(window=5, min_periods=1).mean())
    .fillna(0.33)
)

# Head-to-Head
def create_h2h_effect(row):
    pair = sorted([row['HomeTeam'], row['AwayTeam']])
    return f"{pair[0]}_vs_{pair[1]}"

df['H2H_key'] = df.apply(create_h2h_effect, axis=1)
df['h2h_home_winrate'] = (
    df.groupby('H2H_key')['HomePoints']
    .transform(lambda x: (x.shift(1) == 3).astype(float).expanding(min_periods=1).mean())
    .fillna(0.45)
)

# 6. Merge team_history back into df
print("Merging team statistics to match level...")
features_to_merge = [
    'Date', 'Team', 
    'Form_points_last_5', 'GoalsFor_last_5', 'GoalsAgainst_last_5', 
    'SoT_For_5', 'SoT_Against_5', 'Rest_Days',
    'Cum_Season_Points'
]
team_stats = team_history[features_to_merge].drop_duplicates(subset=['Date', 'Team']).copy()

# Merge for Home Team
df = df.merge(
    team_stats, 
    left_on=['Date', 'HomeTeam'], 
    right_on=['Date', 'Team'], 
    how='left'
).rename(columns={
    'Form_points_last_5': 'Home_Form_5',
    'GoalsFor_last_5': 'Home_GoalsFor_5',
    'GoalsAgainst_last_5': 'Home_GoalsAgainst_5',
    'SoT_For_5': 'Home_SoT_For_5',
    'SoT_Against_5': 'Home_SoT_Against_5',
    'Rest_Days': 'Home_Rest_Days',
    'Cum_Season_Points': 'Home_Cum_Points'
}).drop(columns=['Team'])

# Merge for Away Team
df = df.merge(
    team_stats, 
    left_on=['Date', 'AwayTeam'], 
    right_on=['Date', 'Team'], 
    how='left'
).rename(columns={
    'Form_points_last_5': 'Away_Form_5',
    'GoalsFor_last_5': 'Away_GoalsFor_5',
    'GoalsAgainst_last_5': 'Away_GoalsAgainst_5',
    'SoT_For_5': 'Away_SoT_For_5',
    'SoT_Against_5': 'Away_SoT_Against_5',
    'Rest_Days': 'Away_Rest_Days',
    'Cum_Season_Points': 'Away_Cum_Points'
}).drop(columns=['Team'])

# 7. Differentials
df['Diff_Form_5'] = df['Home_Form_5'] - df['Away_Form_5']
df['Diff_GoalsFor_5'] = df['Home_GoalsFor_5'] - df['Away_GoalsFor_5']
df['Diff_GoalsAgainst_5'] = df['Home_GoalsAgainst_5'] - df['Away_GoalsAgainst_5']
df['Diff_SoT_For_5'] = df['Home_SoT_For_5'] - df['Away_SoT_For_5']
df['Diff_Rest_Days'] = df['Home_Rest_Days'] - df['Away_Rest_Days']
df['Diff_Cum_Points'] = df['Home_Cum_Points'] - df['Away_Cum_Points']

# 8. Save output
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"Features pipeline complete! Saved {df.shape[0]} matches and {df.shape[1]} features to {OUTPUT_PATH}")