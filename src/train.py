import pandas as pd 
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "matches_features.csv"

print("Loading feature data for training...")
df = pd.read_csv(INPUT_PATH, dtype={"Season_code": str}, low_memory=False)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Drop initial rows where rolling stats are NaN
df = df.dropna(subset=['Diff_Form_5']).copy()

# Target mapping for 1X2 classification: 0 = Home Win, 1 = Draw, 2 = Away Win
target_map = {'H': 0, 'D': 1, 'A': 2}
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

# Time-based split: Train on 2000-2023, Test on 2023-2026
train_mask = df["Date"] < '2023-08-01'
test_mask = df['Date'] >= '2023-08-01'

X_train = df.loc[train_mask, feature_cols]
y_train = df.loc[train_mask, 'target']
y_train_home_goals = df.loc[train_mask, 'FTHG']
y_train_away_goals = df.loc[train_mask, 'FTAG']

X_test = df.loc[test_mask, feature_cols]
y_test = df.loc[test_mask, 'target']
y_test_home_goals = df.loc[test_mask, 'FTHG']
y_test_away_goals = df.loc[test_mask, 'FTAG']

print(f"Training set: {X_train.shape[0]} matches")
print(f"Test set:     {X_test.shape[0]} matches")

# =======================================================
# PART 1: 1X2 Outcome Classification Models
# =======================================================
print("\n--- Training 1X2 Outcome Models ---")

# Baseline: Always predict Home Win (Class 0)
majority_preds = np.zeros(len(y_test))
base_acc = accuracy_score(y_test, majority_preds)
print(f"Baseline (Always Home Win) Accuracy: {base_acc:.4f} ({base_acc*100:.1f}%)")

# Model 1: Logistic Regression
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train, y_train)
lr_probs = lr_pipeline.predict_proba(X_test)
print(f"Logistic Regression -> Acc: {accuracy_score(y_test, lr_pipeline.predict(X_test)):.4f} | Log Loss: {log_loss(y_test, lr_probs):.4f}")

# Model 2: XGBoost Classifier
xgb_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_probs = xgb_model.predict_proba(X_test)
print(f"XGBoost Classifier  -> Acc: {accuracy_score(y_test, xgb_model.predict(X_test)):.4f} | Log Loss: {log_loss(y_test, xgb_probs):.4f}")

# Save best 1X2 model
Path("models").mkdir(exist_ok=True)
best_outcome_model = lr_pipeline if log_loss(y_test, lr_probs) < log_loss(y_test, xgb_probs) else xgb_model
joblib.dump(best_outcome_model, "models/model_v1.pkl")

# =======================================================
# PART 2: Poisson Goal / Score Prediction Models
# =======================================================
print("\n--- Training Poisson Exact Score Models ---")

# Home Goals Model
home_goal_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', PoissonRegressor(alpha=1e-4, max_iter=1000))
])
home_goal_model.fit(X_train, y_train_home_goals)

# Away Goals Model
away_goal_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', PoissonRegressor(alpha=1e-4, max_iter=1000))
])
away_goal_model.fit(X_train, y_train_away_goals)

# Evaluate Goals Predictions
pred_h_goals = home_goal_model.predict(X_test)
pred_a_goals = away_goal_model.predict(X_test)
print(f"Home Goals RMSE: {np.sqrt(mean_squared_error(y_test_home_goals, pred_h_goals)):.3f}")
print(f"Away Goals RMSE: {np.sqrt(mean_squared_error(y_test_away_goals, pred_a_goals)):.3f}")

# Save Score Models
score_models = {
    "home_goals_model": home_goal_model,
    "away_goals_model": away_goal_model
}
joblib.dump(score_models, "models/score_model.pkl")

print("\nAll models trained and saved to models/ folder successfully!")