import pandas as pd 
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "matches_features.csv"

df = pd.read_csv(INPUT_PATH, dtype={"Season_code": str}, low_memory=False)


df["Date"] = pd.to_datetime(df["Date"])
df= df.sort_values("Date").reset_index(drop=True)


df = df.dropna(subset=['Diff_Form_5']).copy()


target_map = {'H': 0 , 'D': 1 , 'A' :2}

df['target'] = df['FTR'].map(target_map)





feature_cols = [
    # Elo
    'HomeElo', 'AwayElo', 'EloDiff',
    # Form & Differentials
    'Home_Form_5', 'Away_Form_5', 'Diff_Form_5',
    'Home_GoalsFor_5', 'Away_GoalsFor_5', 'Diff_GoalsFor_5',
    'Home_GoalsAgainst_5', 'Away_GoalsAgainst_5', 'Diff_GoalsAgainst_5',
    # Shots on Target
    'Home_SoT_For_5', 'Away_SoT_For_5', 'Diff_SoT_For_5',
    # Rest Days & Table Position
    'Home_Rest_Days', 'Away_Rest_Days', 'Diff_Rest_Days',
    'Home_Cum_Points', 'Away_Cum_Points', 'Diff_Cum_Points',
    'Home_Table_Pos', 'Away_Table_Pos', 'Diff_Table_Pos',
    # Venue & Head-to-Head
    'Home_venue_effect', 'Away_venue_effect', 'h2h_home_winrate'
]

train_mask = df["Date"] < '2023-08-01'
test_mask = df['Date'] >= '2023-08-01'

X_train = df.loc[train_mask , feature_cols]
y_train = df.loc[train_mask , 'target']


X_test = df.loc[test_mask, feature_cols]
y_test = df.loc[test_mask, 'target']

print(f"Training set: {X_train.shape[0]} matches")
print(f"Test set:     {X_test.shape[0]} matches")