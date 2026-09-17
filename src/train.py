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



# ==========================================
# ADD EVERYTHING BELOW THIS LINE:
# ==========================================

from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# 1. Baseline: Always predict Home Win (Class 0)
majority_preds = np.zeros(len(y_test))
base_acc = accuracy_score(y_test, majority_preds)
print(f"\n--- Baseline (Always Home Win) ---")
print(f"Accuracy: {base_acc:.4f} ({base_acc*100:.1f}%)")

# 2. Model 1: Logistic Regression
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
lr_probs = lr_pipeline.predict_proba(X_test)

print("\n--- 1. Logistic Regression ---")
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
print(f"Log Loss: {log_loss(y_test, lr_probs):.4f}")

# 3. Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)

print("\n--- 2. Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(f"Log Loss: {log_loss(y_test, rf_probs):.4f}")

# 4. Model 3: XGBoost
xgb_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)

print("\n--- 3. XGBoost ---")
print(f"Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
print(f"Log Loss: {log_loss(y_test, xgb_probs):.4f}")

# 5. Save the best model
Path("models").mkdir(exist_ok=True)
best_model = lr_pipeline if log_loss(y_test, lr_probs) < log_loss(y_test, xgb_probs) else xgb_model
joblib.dump(best_model, "models/model_v1.pkl")
print("\nSaved best model to models/model_v1.pkl!")