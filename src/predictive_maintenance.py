"""
Predictive Maintenance using ML — CMAPSS FD001
Author: Tharoon K
Institution: Madras Institute of Technology, Chennai
Dataset: NASA CMAPSS Turbofan Engine Degradation Dataset
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib

# ── Configuration ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_PATH = PROJECT_ROOT / "results"
MODELS_PATH = PROJECT_ROOT / "models"

RESULTS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)

RUL_CAP     = 125
WINDOW      = 5
N_TREES     = 150
MAX_DEPTH   = 15
RANDOM_SEED = 42

COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
DROP_COLS   = ['op3', 's1', 's5', 's6', 's10', 's16', 's17', 's18', 's19']
FEATURES    = ['s2', 's3', 's4', 's7', 's8', 's9',
               's11', 's12', 's13', 's14', 's15', 's20', 's21']

# ── Step 1: Load data ────────────────────────────────────────────
def load_data(filename):
    df = pd.read_csv(
        DATA_PATH / filename,
        sep=r'\s+', header=None, names=COLS, engine='python'
    )
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    return df

print("Step 1: Loading data...")
train = load_data('train_FD001.txt')
test  = load_data('test_FD001.txt')
print(f"  Train: {train.shape} | Test: {test.shape}")

# ── Step 2: Calculate RUL ────────────────────────────────────────
def add_rul_train(df):
    max_cycle = df.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max_cycle']
    df = df.merge(max_cycle, on='unit')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns=['max_cycle'], inplace=True)
    return df

def add_rul_test(df, rul_file):
    rul = pd.read_csv(DATA_PATH / rul_file, header=None, names=['RUL'])
    rul['unit'] = rul.index + 1
    last = df.groupby('unit')['cycle'].max().reset_index()
    last.columns = ['unit', 'last_cycle']
    df = df.merge(last, on='unit').merge(rul, on='unit')
    df['RUL'] = df['RUL'] + (df['last_cycle'] - df['cycle'])
    df.drop(columns=['last_cycle'], inplace=True)
    return df

print("Step 2: Calculating RUL...")
train = add_rul_train(train)
test  = add_rul_test(test, 'RUL_FD001.txt')

# ── Step 3: Rolling window features ─────────────────────────────
def add_rolling(df, sensors, window):
    for s in sensors:
        df[f'{s}_mean{window}'] = df.groupby('unit')[s].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'{s}_std{window}'] = df.groupby('unit')[s].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
    return df

print("Step 3: Adding rolling features...")
train = add_rolling(train, FEATURES, WINDOW)
test  = add_rolling(test,  FEATURES, WINDOW)

ALL_FEATURES = (FEATURES +
                [f'{s}_mean{WINDOW}' for s in FEATURES] +
                [f'{s}_std{WINDOW}'  for s in FEATURES])
print(f"  Total features: {len(ALL_FEATURES)}")

# ── Step 4: Prepare X and y ──────────────────────────────────────
print("Step 4: Preparing features...")
X_train = train[ALL_FEATURES]
y_train = train['RUL'].clip(upper=RUL_CAP)

X_test  = test[ALL_FEATURES]
y_test  = test['RUL'].clip(upper=RUL_CAP)

scaler      = MinMaxScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

# ── Step 5: Train model ──────────────────────────────────────────
print("Step 5: Training Random Forest...")
model = RandomForestRegressor(
    n_estimators=N_TREES,
    max_depth=MAX_DEPTH,
    min_samples_leaf=2,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
model.fit(X_train_sc, y_train)
print("  Training complete.")

# ── Step 6: Evaluate ─────────────────────────────────────────────
print("Step 6: Evaluating...")
y_pred = model.predict(X_test_sc)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"  RMSE : {rmse:.2f} cycles")
print(f"  MAE  : {mae:.2f} cycles")
print(f"  R²   : {r2:.4f}")
print(f"{'='*40}")

# ── Step 7: Save model ───────────────────────────────────────────
print("\nStep 7: Saving model...")
joblib.dump(model,  MODELS_PATH / 'rf_model.pkl')
joblib.dump(scaler, MODELS_PATH / 'scaler.pkl')
print("  Model saved as rf_model.pkl")
print("  Scaler saved as scaler.pkl")

# ── Step 8: Plots ────────────────────────────────────────────────
print("\nStep 8: Generating plots...")

# Plot 1 — Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', s=10)
plt.plot([0, RUL_CAP], [0, RUL_CAP], color='red', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual RUL (cycles)')
plt.ylabel('Predicted RUL (cycles)')
plt.title('Random Forest — Actual vs Predicted RUL')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_PATH / 'actual_vs_predicted.png', dpi=150)
plt.show()

# Plot 2 — Feature importance
importances = pd.Series(model.feature_importances_, index=ALL_FEATURES)
top15 = importances.sort_values(ascending=False).head(15).sort_values()
plt.figure(figsize=(8, 6))
top15.plot(kind='barh', color='steelblue')
plt.xlabel('Importance score')
plt.title('Top 15 features driving model predictions')
plt.tight_layout()
plt.savefig(RESULTS_PATH / 'feature_importance.png', dpi=150)
plt.show()

# Plot 3 — Single engine RUL tracking
unit_id  = 5
test_u   = test[test['unit'] == unit_id].copy()
test_u_sc = scaler.transform(test_u[ALL_FEATURES])
test_u['predicted_RUL'] = model.predict(test_u_sc)

plt.figure(figsize=(8, 4))
plt.plot(test_u['cycle'], test_u['RUL'].clip(upper=RUL_CAP),
         label='Actual RUL', color='steelblue')
plt.plot(test_u['cycle'], test_u['predicted_RUL'],
         label='Predicted RUL', color='orange', linestyle='--')
plt.xlabel('Cycle')
plt.ylabel('RUL')
plt.title(f'Engine Unit {unit_id} — RUL tracking over time')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_PATH / 'rul_tracking.png', dpi=150)
plt.show()

print("\nDone. Models and plots saved to the project folders.")
