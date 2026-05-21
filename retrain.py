"""
Retrain all PaySim fraud detection models from scratch.
Generates synthetic PaySim-like data, trains all models,
saves pkls and SHAP plots to /models folder.
Run: python retrain.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve)
from xgboost import XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────
# 1. Generate synthetic PaySim-like dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Generating synthetic PaySim-like dataset...")
print("=" * 60)

N_LEGIT = 180_000
N_FRAUD = 2_000  # ~1.1% fraud rate, mirrors real PaySim

tx_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
fraud_types = ["TRANSFER", "CASH_OUT"]

rows = []

# Legit transactions
for _ in range(N_LEGIT):
    tx_type = np.random.choice(tx_types, p=[0.35, 0.20, 0.25, 0.15, 0.05])
    step = np.random.randint(1, 744)
    oldbal = np.random.exponential(scale=50000) + 100
    amount = np.random.exponential(scale=10000) + 10
    amount = min(amount, oldbal * 0.95)

    if tx_type in ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]:
        newbal_orig = max(0, oldbal - amount)
    else:
        newbal_orig = oldbal + amount

    oldbal_dest = np.random.exponential(scale=20000)
    newbal_dest = oldbal_dest + amount

    rows.append([step, tx_type, round(amount, 2), round(oldbal, 2),
                 round(newbal_orig, 2), round(oldbal_dest, 2),
                 round(newbal_dest, 2), 0])

# Fraud transactions — mimic PaySim fraud patterns
for _ in range(N_FRAUD):
    tx_type = np.random.choice(fraud_types, p=[0.5, 0.5])
    step = np.random.randint(1, 744)

    # Pattern 1: Full balance drain
    if np.random.random() < 0.6:
        oldbal = np.random.exponential(scale=200000) + 1000
        amount = oldbal * np.random.uniform(0.9, 1.0)
        newbal_orig = max(0.0, oldbal - amount)
        oldbal_dest = 0.0
        newbal_dest = amount

    # Pattern 2: Large amount, destination was zero
    else:
        oldbal = np.random.exponential(scale=100000) + 5000
        amount = np.random.uniform(50000, 500000)
        amount = min(amount, oldbal)
        newbal_orig = max(0.0, oldbal - amount)
        oldbal_dest = 0.0
        newbal_dest = amount

    rows.append([step, tx_type, round(amount, 2), round(oldbal, 2),
                 round(newbal_orig, 2), round(oldbal_dest, 2),
                 round(newbal_dest, 2), 1])

cols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "isFraud"]

df = pd.DataFrame(rows, columns=cols).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"  Dataset shape: {df.shape}")
print(f"  Fraud rate: {df['isFraud'].mean():.2%}")
print(f"  Transaction types:\n{df['type'].value_counts().to_string()}")

# ─────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────
print("\nSTEP 2: Feature engineering...")

df["orig_balance_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]

FEATURES = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "orig_balance_delta", "dest_balance_delta"]
TARGET = "isFraud"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
print(f"  Train fraud: {y_train.sum()} | Test fraud: {y_test.sum()}")

# ─────────────────────────────────────────────
# 3. Preprocessor
# ─────────────────────────────────────────────
print("\nSTEP 3: Building preprocessor...")

numeric_features = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
                    "oldbalanceDest", "newbalanceDest",
                    "orig_balance_delta", "dest_balance_delta"]
categorical_features = ["type"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
])

# Fit and save
preprocessor.fit(X_train)
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocess.pkl"))
print("  Saved preprocess.pkl")

X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)

# ─────────────────────────────────────────────
# 4. Train all models
# ─────────────────────────────────────────────
print("\nSTEP 4: Training models...")

results = {}

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    metrics = {
        "accuracy":  round(accuracy_score(y_te, y_pred), 6),
        "precision": round(precision_score(y_te, y_pred, zero_division=0), 6),
        "recall":    round(recall_score(y_te, y_pred, zero_division=0), 6),
        "f1":        round(f1_score(y_te, y_pred, zero_division=0), 6),
        "roc_auc":   round(roc_auc_score(y_te, y_prob), 6),
        "pr_auc":    round(average_precision_score(y_te, y_prob), 6),
    }
    print(f"  {name:30s} F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}")
    return metrics, y_prob

# Logistic Regression
print("\n  Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
results["lr"], lr_probs = evaluate("Logistic Regression", lr, X_train_t, y_train, X_test_t, y_test)
joblib.dump(lr, os.path.join(MODELS_DIR, "lr.pkl"))

# Decision Tree
print("  Training Decision Tree...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
best_depth, best_score = 5, 0
for depth in [3, 5, 7, 10, 15]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE, class_weight="balanced")
    score = cross_val_score(dt, X_train_t, y_train, cv=cv, scoring="f1").mean()
    if score > best_score:
        best_score, best_depth = score, depth
tree = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE, class_weight="balanced")
results["tree"], tree_probs = evaluate(f"Decision Tree (depth={best_depth})", tree, X_train_t, y_train, X_test_t, y_test)
joblib.dump(tree, os.path.join(MODELS_DIR, "tree.pkl"))

# Random Forest
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2,
                             random_state=RANDOM_STATE, class_weight="balanced",
                             n_jobs=-1)
results["rf"], rf_probs = evaluate("Random Forest", rf, X_train_t, y_train, X_test_t, y_test)
joblib.dump(rf, os.path.join(MODELS_DIR, "rf.pkl"))

# XGBoost
print("  Training XGBoost...")
scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     scale_pos_weight=scale_pos,
                     random_state=RANDOM_STATE, eval_metric="aucpr",
                     verbosity=0)
results["xgb"], xgb_probs = evaluate("XGBoost", xgb, X_train_t, y_train, X_test_t, y_test)
joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb.pkl"))

# Save best params
best_params = {"decision_tree_best_depth": best_depth, "xgb_scale_pos_weight": scale_pos}
with open(os.path.join(MODELS_DIR, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

# ─────────────────────────────────────────────
# 5. ROC and PR curves
# ─────────────────────────────────────────────
print("\nSTEP 5: Saving ROC and PR curves...")

model_plot_data = {
    "logistic_regression": (lr_probs,   "#6366f1"),
    "decision_tree":       (tree_probs, "#f59e0b"),
    "random_forest":       (rf_probs,   "#22c55e"),
    "xgboost":             (xgb_probs,  "#3b82f6"),
}

for name, (probs, color) in model_plot_data.items():
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#0f1629")
    ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0,1],[0,1], "gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate", color="white", fontsize=10)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=10)
    ax.set_title(f"ROC — {name.replace('_',' ').title()}", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#1e3a5f")
    ax.legend(facecolor="#0f1629", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f"{name}_roc_curve.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#0f1629")
    ax.plot(rec, prec, color=color, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall", color="white", fontsize=10)
    ax.set_ylabel("Precision", color="white", fontsize=10)
    ax.set_title(f"PR Curve — {name.replace('_',' ').title()}", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#1e3a5f")
    ax.legend(facecolor="#0f1629", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f"{name}_pr_curve.png"), dpi=120, bbox_inches="tight")
    plt.close()

print("  Saved all ROC and PR curve plots.")

# ─────────────────────────────────────────────
# 6. SHAP explainability
# ─────────────────────────────────────────────
print("\nSTEP 6: Generating SHAP plots...")

if SHAP_AVAILABLE:
    try:
        # Use a sample for speed
        sample_idx = np.random.choice(len(X_test_t), min(500, len(X_test_t)), replace=False)
        X_sample = X_test_t[sample_idx]

        # Feature names
        num_names = numeric_features
        cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features))
        feature_names = num_names + cat_names

        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_sample)

        # Summary plot
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#0a0f1e")
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          show=False, plot_size=None)
        plt.title("SHAP Summary Plot", color="white", fontsize=12, pad=10)
        plt.gcf().set_facecolor("#0a0f1e")
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, "shap_summary.png"), dpi=120,
                    bbox_inches="tight", facecolor="#0a0f1e")
        plt.close()

        # Bar plot
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0a0f1e")
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          plot_type="bar", show=False, plot_size=None)
        plt.title("SHAP Feature Importance", color="white", fontsize=12, pad=10)
        plt.gcf().set_facecolor("#0a0f1e")
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, "shap_bar.png"), dpi=120,
                    bbox_inches="tight", facecolor="#0a0f1e")
        plt.close()

        # Waterfall for a single fraud transaction
        fraud_indices = np.where(y_test.values[sample_idx] == 1)[0]
        if len(fraud_indices) > 0:
            idx = fraud_indices[0]
            shap_exp = shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=X_sample[idx],
                feature_names=feature_names
            )
            fig = plt.figure(figsize=(9, 5))
            fig.patch.set_facecolor("#0a0f1e")
            shap.plots.waterfall(shap_exp, show=False, max_display=10)
            plt.gcf().set_facecolor("#0a0f1e")
            plt.tight_layout()
            plt.savefig(os.path.join(MODELS_DIR, "shap_waterfall.png"), dpi=120,
                        bbox_inches="tight", facecolor="#0a0f1e")
            plt.close()

        print("  Saved shap_summary.png, shap_bar.png, shap_waterfall.png")
    except Exception as ex:
        print(f"  SHAP plotting failed: {ex} — continuing without SHAP plots.")
else:
    print("  SHAP not available — skipping.")

# ─────────────────────────────────────────────
# 7. EDA plots
# ─────────────────────────────────────────────
print("\nSTEP 7: Saving EDA plots...")

# Class distribution
fig, ax = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor("#0a0f1e")
ax.set_facecolor("#0f1629")
counts = df["isFraud"].value_counts()
bars = ax.bar(["Legit", "Fraud"], counts.values, color=["#22c55e", "#ef4444"], width=0.5)
ax.set_title("Class Distribution", color="white", fontsize=12)
ax.tick_params(colors="white")
ax.spines[:].set_color("#1e3a5f")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f"{val:,}", ha="center", color="white", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "eda_class_distribution.png"), dpi=120,
            bbox_inches="tight", facecolor="#0a0f1e")
plt.close()

# Transaction type distribution
fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor("#0a0f1e")
ax.set_facecolor("#0f1629")
type_counts = df["type"].value_counts()
colors_bar = ["#3b82f6","#6366f1","#8b5cf6","#a78bfa","#c4b5fd"]
ax.bar(type_counts.index, type_counts.values, color=colors_bar, width=0.6)
ax.set_title("Transaction Types", color="white", fontsize=12)
ax.tick_params(colors="white")
ax.spines[:].set_color("#1e3a5f")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "eda_transaction_type.png"), dpi=120,
            bbox_inches="tight", facecolor="#0a0f1e")
plt.close()

print("  Saved EDA plots.")

# ─────────────────────────────────────────────
# 8. Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL DONE — models retrained and saved to /models")
print("=" * 60)
print("\nModel results summary:")
print(f"{'Model':<25} {'F1':>8} {'ROC-AUC':>10} {'PR-AUC':>10}")
print("-" * 55)
model_names = {"lr": "Logistic Regression", "tree": "Decision Tree",
               "rf": "Random Forest", "xgb": "XGBoost"}
for key, name in model_names.items():
    r = results[key]
    print(f"{name:<25} {r['f1']:>8.4f} {r['roc_auc']:>10.4f} {r['pr_auc']:>10.4f}")

print("\nFiles saved to /models:")
for f in sorted(os.listdir(MODELS_DIR)):
    print(f"  {f}")

print("\nNow run:")
print('  python -m streamlit run streamlit_app.py')
