# Premier League Match Predictor — End-to-End Workflow

A project designed to teach you real data cleaning, transformation, and ML production skills using football data.

---

## 0. Define the Problem Before Touching Data

Be specific about what you're predicting — it changes everything downstream.

- **Classification (simplest, recommended start):** Home Win / Draw / Away Win (3-class)
- **Alternative classification:** Over/Under 2.5 goals, Both Teams to Score
- **Regression (harder):** Exact goals scored by each team

**Recommendation:** Start with 3-class match outcome prediction. It's the classic problem, has clean labels, and lets you benchmark against bookmaker odds.

---

## 1. Data Sources

| Source | What you get | Notes |
|---|---|---|
| **football-data.co.uk** | Free CSVs, match results 1993–present, odds, shots, cards, corners | Best starting point — clean-ish, no API key needed |
| **Kaggle (English Premier League datasets)** | Pre-packaged historical match data | Good for quick starts, less current |
| **API-Football (api-football.com)** | Live/current fixtures, stats, lineups via REST API | Free tier is rate-limited; needed for production/live predictions |
| **FBref.com** | Advanced stats: xG, xA, progressive passes, pressures | Requires scraping (check their terms); goldmine for features |
| **Understat.com** | xG data specifically | Also scrape-only |
| **Football-Data.org API** | Fixtures, standings, basic stats | Free tier available, good for production odds/fixtures |

**For learning cleaning/transformation specifically:** pull raw CSVs from football-data.co.uk across multiple seasons — the team naming inconsistencies, missing columns across years, and format changes across seasons are exactly the mess you want to practice on.

---

## 2. Environment Setup

```bash
mkdir pl-predictor && cd pl-predictor
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

pip install pandas numpy scikit-learn matplotlib seaborn jupyter \
            requests beautifulsoup4 xgboost lightgbm \
            fastapi uvicorn joblib python-dotenv
```

Project structure:
```
pl-predictor/
├── data/
│   ├── raw/            # untouched downloads
│   ├── interim/        # partially cleaned
│   └── processed/      # model-ready
├── notebooks/           # exploration
├── src/
│   ├── ingest.py
│   ├── clean.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── models/
├── api/
│   └── main.py
└── requirements.txt
```

---

## 3. Data Ingestion

- Download 8–10 seasons of CSVs from football-data.co.uk (2015/16 onward is plenty).
- Write `src/ingest.py` to loop through season files and concatenate them, tagging each row with its season.
- **Immediately save the raw concatenated file** to `data/raw/` before any cleaning — never overwrite raw data.

```python
import pandas as pd
import glob

files = glob.glob("data/raw/season_*.csv")
dfs = [pd.read_csv(f) for f in files]
raw = pd.concat(dfs, ignore_index=True)
raw.to_csv("data/interim/all_seasons_raw.csv", index=False)
```

---

## 4. Data Cleaning (the core learning phase)

This is where most of your learning will happen. Work through each systematically and **document what you did and why** — that documentation is 80% of the value of this exercise.

### 4.1 Inspect first
```python
df.info()
df.isnull().sum()
df.describe()
df.duplicated().sum()
```

### 4.2 Team name standardization
Different seasons/sources spell teams differently: "Man United" vs "Manchester United" vs "Man Utd". This will silently break every join and feature you build later.
```python
name_map = {
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham Hotspur",
    # build this out fully — check unique values first
}
df["HomeTeam"] = df["HomeTeam"].replace(name_map)
df["AwayTeam"] = df["AwayTeam"].replace(name_map)
```
Always run `df["HomeTeam"].unique()` and eyeball it before and after.

### 4.3 Handle missing values
- Older seasons often lack shot/xG data — decide: drop the column, drop pre-2010 rows, or impute.
- **Never impute the target** (match result) — drop rows missing it.
- For stats like shots/corners, consider median imputation *per season* (not global) since playing styles/eras shift.

### 4.4 Fix data types
- Dates: `pd.to_datetime(df["Date"], dayfirst=True)` (UK format trap — this bites everyone)
- Ensure goal columns are `int`, odds are `float`
- Categorical columns (`HomeTeam`, `Referee`) → `category` dtype for memory efficiency

### 4.5 Remove duplicates and structural errors
- Duplicate fixtures (same teams, same date)
- Impossible values (negative goals, matches with no result)

### 4.6 Outlier review (don't blindly drop)
- 9-0 scorelines are real football, not errors — validate against reality before removing anything.

**Deliverable of this stage:** a clean `data/processed/matches_clean.csv` and a short `CLEANING_LOG.md` describing every decision you made.

---

## 5. Feature Engineering & Transformation

This is the "transformation" half of your goal — turning raw match rows into features a model can learn from.

### 5.1 Critical rule: no future leakage
For any match, you can only use information available **before kickoff**. This means rolling stats must be *shifted* — never include the match itself in its own rolling average.

### 5.2 Core features to build
- **Form:** points from last 5 matches (rolling, shifted), separately for home/away
- **Goal statistics:** rolling average goals scored/conceded (last 5, last 10)
- **Head-to-head:** historical result tendency between the two specific teams
- **Home/away splits:** a team's home-only win rate vs away-only win rate
- **Rest days:** days since each team's last match (fixture congestion matters)
- **Elo rating:** build a simple incrementally-updated Elo score per team (very high-value feature, and a great coding exercise)
- **Table position / points at time of match** (not end of season — this must be as-of-date)

```python
df = df.sort_values("Date")
df["HomeTeamPoints_Last5"] = (
    df.groupby("HomeTeam")["HomePoints"]
      .transform(lambda x: x.shift(1).rolling(5).mean())
)
```

### 5.3 Encoding
- Team identity: target encoding or embeddings if using neural nets; one-hot is fine for tree models but gets wide with 20+ teams
- Categorical referee/venue: frequency or target encoding

### 5.4 Scaling
- Tree-based models (XGBoost, Random Forest) don't need scaling.
- If you try logistic regression or SVM, use `StandardScaler` — fit on train only, transform test.

### 5.5 Train/test split — **time-based, not random**
Never randomly shuffle match data. Use a chronological split (e.g., train on 2015–2023, test on 2023–2024) or `TimeSeriesSplit` for cross-validation. Random shuffling leaks future form into training and will give you a fake, inflated accuracy.

---

## 6. Exploratory Data Analysis

Before modeling, plot:
- Home win % vs away win % vs draw % (base rates — your model's accuracy floor)
- Goals scored distribution by team/season
- Correlation heatmap of engineered features vs outcome
- Elo rating vs actual result accuracy (sanity check)

---

## 7. Modeling

### 7.1 Baselines first (always)
- Predict "Home Win" every time (home advantage is real, ~45% base rate)
- Predict using bookmaker implied probabilities from the odds columns — this is your real benchmark, not just accuracy

### 7.2 Models to try, in order
1. Logistic Regression (multinomial) — interpretable baseline
2. Random Forest Classifier
3. XGBoost / LightGBM — usually best performer for this kind of tabular data
4. (Optional, advanced) Poisson regression modeling home/away goals separately, then deriving win/draw/loss probabilities

### 7.3 Evaluation metrics
- Accuracy (but know its limits — draws are hard, ~25% of matches, models often just avoid predicting them)
- **Log loss** — better metric since you want calibrated probabilities, not just the top class
- **Brier score** — measures probability calibration
- Confusion matrix — check if the model just always predicts "Home Win"
- Compare your log loss against bookmaker odds' implied log loss — beating the bookmakers is genuinely hard and a meaningful benchmark

### 7.4 Cross-validation
Use `TimeSeriesSplit` from scikit-learn, not standard k-fold, to respect chronology.

---

## 8. From Notebook to Production Code

- Move logic out of Jupyter into `src/` modules once it's stable
- Use `joblib.dump(model, "models/model_v1.pkl")` to persist the trained model, plus the fitted encoders/scalers
- Write a `predict.py` that takes upcoming fixtures, builds the same features (using the same pipeline — this is critical, a mismatch here is the #1 source of production bugs), and outputs probabilities

**Build a proper sklearn `Pipeline`** combining preprocessing + model so train-time and predict-time transformations can never drift apart:
```python
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier())
])
```

---

## 9. Serving Predictions (Production)

### Option A — Simple API (recommended starting point)
```python
# api/main.py
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/model_v1.pkl")

@app.post("/predict")
def predict(home_team: str, away_team: str):
    features = build_features(home_team, away_team)  # your feature pipeline
    proba = model.predict_proba(features)[0]
    return {"home_win": proba[0], "draw": proba[1], "away_win": proba[2]}
```
Run with `uvicorn api.main:app --reload`.

### Option B — Dashboard
Use **Streamlit** to build a simple UI where you pick two teams and see predicted probabilities — faster to build than a full API and great for demoing the project.

### Automating fresh data
- Write a scheduled script (cron job, or GitHub Actions on a schedule) that pulls new results weekly from your data source, appends to your dataset, and retrains
- Keep model versions (`model_v1.pkl`, `model_v2.pkl`) so you can roll back

---

## 10. Monitoring & Retraining

- Track prediction accuracy/log loss against actual results week over week
- Retrain periodically (e.g., every international break or monthly) as new match data comes in
- Watch for **model drift**: a manager change, promoted/relegated teams, or transfer window can shift a team's true strength faster than rolling averages capture

---

## Suggested Build Order (if you want a checklist)

1. Download 8 seasons of CSVs
2. Concatenate + inspect raw data
3. Clean team names, dates, missing values → write CLEANING_LOG.md
4. Build 5–8 engineered features (start with form, Elo, home/away splits)
5. Chronological train/test split
6. Baseline model (bookmaker odds comparison)
7. Train logistic regression, then XGBoost
8. Evaluate with log loss + Brier score, not just accuracy
9. Wrap in a Pipeline, save with joblib
10. Build a FastAPI endpoint or Streamlit dashboard
11. Automate weekly data refresh + retraining

---

## Where You'll Learn the Most

Honestly, the biggest learning moments in a project like this are usually:
- Realizing your "great" model was leaking future data into training (this happens to almost everyone the first time)
- The team-name-matching mess across data sources
- Discovering how hard it is to beat bookmaker odds, which recalibrates what a "good" accuracy number actually looks like

Good luck — this is a genuinely solid project for a portfolio, since it demonstrates the full pipeline rather than just a Kaggle notebook.
