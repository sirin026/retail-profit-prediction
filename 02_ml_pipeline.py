"""
Retail Profit Prediction — Machine Learning Pipeline
Dataset: US Superstore Sales Data (2014–2017)
Author: Siri Namala

Models: Linear Regression, SVR, Random Forest, KNN, Neural Network (MLP)

Usage:
    1. Download Superstore dataset from Kaggle:
       https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
    2. Place 'Sample - Superstore.csv' in this folder
    3. Run: python 02_ml_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ── STEP 1: LOAD & CLEAN DATA ─────────────────────────────────────────────────
print("Step 1: Loading data...")
df = pd.read_csv('Sample - Superstore.csv', encoding='latin-1')

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

# Parse dates
df['order_date'] = pd.to_datetime(df['order_date'])
df['ship_date']  = pd.to_datetime(df['ship_date'])

# Derive features
df['shipping_days'] = (df['ship_date'] - df['order_date']).dt.days
df['month']         = df['order_date'].dt.month
df['year']          = df['order_date'].dt.year
df['profit_margin'] = df['profit'] / df['sales']

print(f"  Loaded {len(df):,} orders | {(df['profit'] < 0).sum():,} loss orders ({(df['profit']<0).mean()*100:.1f}%)")
print(f"  Profit std dev: ${df['profit'].std():.2f}")

# ── STEP 2: FEATURE ENGINEERING ───────────────────────────────────────────────
print("\nStep 2: Feature engineering...")

# Drop high-cardinality / leakage columns
drop_cols = ['row_id', 'order_id', 'customer_id', 'customer_name', 'product_id',
             'product_name', 'city', 'postal_code', 'country',
             'order_date', 'ship_date', 'profit_margin']
df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Encode categoricals
le = LabelEncoder()
cat_cols = df_model.select_dtypes(include='object').columns
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Features & target
X = df_model.drop(columns=['profit'])
y = df_model['profit']

print(f"  Features: {list(X.columns)}")
print(f"  X shape: {X.shape} | y shape: {y.shape}")

# Train-test split: 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale features (required for SVR, KNN, MLP)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── STEP 3: TRAIN MODELS ──────────────────────────────────────────────────────
print("\nStep 3: Training models...")

models = {
    'Linear Regression': LinearRegression(),
    'SVR':               SVR(kernel='rbf', C=1.0, epsilon=0.2),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
    'KNN':               KNeighborsRegressor(n_neighbors=5),
    'Neural Network':    MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

results = {}
for name, model in models.items():
    # Use scaled data for SVR, KNN, MLP; raw for tree-based + linear
    use_scaled = name in ['SVR', 'KNN', 'Neural Network']
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test

    model.fit(Xtr, y_train)
    y_pred_train = model.predict(Xtr)
    y_pred_test  = model.predict(Xte)

    r2_train  = r2_score(y_train, y_pred_train)
    r2_test   = r2_score(y_test,  y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test  = mean_squared_error(y_test,  y_pred_test)

    results[name] = {
        'r2_train': round(r2_train, 4),
        'r2_test':  round(r2_test, 4),
        'mse_train': round(mse_train, 2),
        'mse_test':  round(mse_test, 2)
    }
    print(f"  {name:20s} | Train R²={r2_train:.4f} | Test R²={r2_test:.4f} | Test MSE={mse_test:,.0f}")

# ── STEP 4: RESULTS SUMMARY ───────────────────────────────────────────────────
print("\n" + "="*65)
print("  Model Performance Summary")
print("="*65)
results_df = pd.DataFrame(results).T
print(results_df.to_string())

best_model = results_df['r2_test'].idxmax()
print(f"\n  Best model by Test R²: {best_model} (R²={results_df.loc[best_model,'r2_test']})")
print(f"  Note: R² ~0.15 indicates profit is driven by factors not in the")
print(f"        order data alone (e.g., cost of goods, competitor pricing).")

# ── STEP 5: FEATURE IMPORTANCE (Random Forest) ────────────────────────────────
print("\nStep 5: Feature importance (Random Forest)...")
rf_model = models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(10)
print(importances.to_string())

# ── STEP 6: GENERATE CHARTS ───────────────────────────────────────────────────
print("\nStep 6: Saving charts...")

NAVY, TEAL, ORANGE, RED, GREEN = '#1F4E79','#2E86AB','#E07B39','#C0392B','#27AE60'
GRID = '#f0f4f8'
model_names = list(results.keys())
short_names = ['Linear\nReg', 'SVR', 'Random\nForest', 'KNN', 'Neural\nNet']

# Chart 1: Model R² comparison
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(model_names)); w = 0.35
r2_tr = [results[m]['r2_train'] for m in model_names]
r2_te = [results[m]['r2_test']  for m in model_names]
ax.bar(x-w/2, r2_tr, w, label='Train R²', color=TEAL,   edgecolor='white')
ax.bar(x+w/2, r2_te, w, label='Test R²',  color=ORANGE, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(short_names)
ax.set_ylabel('R² Score'); ax.set_title('Model Performance: R² (Train vs Test)', fontweight='bold')
ax.legend(); ax.set_facecolor(GRID); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('chart1_model_performance.png', dpi=150, bbox_inches='tight')
plt.close(); print("  chart1 saved.")

# Chart 2: Feature importance (RF)
fig, ax = plt.subplots(figsize=(10, 5))
importances.sort_values().plot.barh(ax=ax, color=TEAL, edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontweight='bold', fontsize=13)
ax.set_xlabel('Importance Score'); ax.set_facecolor(GRID); ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False); ax.xaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('chart2_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close(); print("  chart2 saved.")

# Chart 3: Profit distribution
fig, ax = plt.subplots(figsize=(10, 5))
counts, bins, patches = ax.hist(y, bins=60, edgecolor='white', linewidth=0.4)
for patch, left in zip(patches, bins[:-1]):
    patch.set_facecolor(RED if left < 0 else TEAL)
ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
ax.set_title('Profit Distribution — Superstore Orders', fontweight='bold', fontsize=13)
ax.set_xlabel('Profit (USD)'); ax.set_ylabel('Number of Orders')
ax.set_facecolor(GRID); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('chart3_profit_distribution.png', dpi=150, bbox_inches='tight')
plt.close(); print("  chart3 saved.")

# Chart 4: Actual vs Predicted (best model)
best = models[best_model]
use_scaled = best_model in ['SVR', 'KNN', 'Neural Network']
Xte_use = X_test_sc if use_scaled else X_test
y_pred = best.predict(Xte_use)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.3, s=12, color=TEAL)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
ax.set_xlabel('Actual Profit'); ax.set_ylabel('Predicted Profit')
ax.set_title(f'Actual vs. Predicted Profit — {best_model}\n(Test R² = {results[best_model]["r2_test"]:.4f})', fontweight='bold', fontsize=12)
ax.legend(); ax.set_facecolor(GRID); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('chart4_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close(); print("  chart4 saved.")

print("\n✓ Pipeline complete! All charts saved.")
