import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import shap
from tensorflow import keras

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="PaySim Fraud Risk Console", layout="wide")

MODELS_DIR = "models"

@st.cache_resource
def load_assets():
    """Load models + config files once."""
    lr = joblib.load(os.path.join(MODELS_DIR, "lr.pkl"))
    tree = joblib.load(os.path.join(MODELS_DIR, "tree.pkl"))
    rf = joblib.load(os.path.join(MODELS_DIR, "rf.pkl"))
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgb.pkl"))
    preprocess = joblib.load(os.path.join(MODELS_DIR, "preprocess.pkl"))
    mlp = keras.models.load_model(os.path.join(MODELS_DIR, "mlp.keras"))

    # Optional files
    comparison_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    best_params_path = os.path.join(MODELS_DIR, "best_params.json")

    results_df = pd.read_csv(comparison_path) if os.path.exists(comparison_path) else None
    best_params = None
    if os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            best_params = json.load(f)

    return lr, tree, rf, xgb, preprocess, mlp, results_df, best_params

lr, tree, rf, xgb, preprocess, mlp, results_df, best_params = load_assets()

# -----------------------------
# Helper functions
# -----------------------------
def build_input_df(step, tx_type, amount, old_org, new_org, old_dest, new_dest):
    """Build a single-row dataframe matching training features (excluding dropped cols)."""
    orig_delta = old_org - new_org
    dest_delta = new_dest - old_dest

    row = {
        "step": step,
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "orig_balance_delta": orig_delta,
        "dest_balance_delta": dest_delta,
    }
    return pd.DataFrame([row])

def predict_with_model(model_name, X_row_df):
    """Return (pred_label, pred_prob)."""
    if model_name == "MLP (Keras)":
        Xp = preprocess.transform(X_row_df)
        if hasattr(Xp, "toarray"):
            Xp = Xp.toarray()
        prob = float(mlp.predict(Xp, verbose=0).ravel()[0])
        label = int(prob >= 0.5)
        return label, prob

    model_map = {
        "Logistic Regression": lr,
        "Decision Tree (CV)": tree,
        "Random Forest (CV)": rf,
        "XGBoost (CV)": xgb,
    }

    model = model_map[model_name]
    prob = float(model.predict_proba(X_row_df)[:, 1][0])
    label = int(prob >= 0.5)
    return label, prob

def shap_waterfall_for_input(X_row_df, max_display=15):
    """
    Generate SHAP waterfall for the custom input using XGBoost (best model).
    We explain the preprocessed features because pipeline uses one-hot encoding.
    """
    xgb_model = xgb.named_steps["model"]

    Xp = preprocess.transform(X_row_df)
    if hasattr(Xp, "toarray"):
        Xp_dense = Xp.toarray()
    else:
        Xp_dense = Xp

    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(Xp_dense)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0] if len(np.atleast_1d(expected_value)) == 1 else expected_value[1]

    exp = shap.Explanation(
        values=shap_vals[0],
        base_values=expected_value,
        data=Xp_dense[0]
    )

    fig = plt.figure()
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    plt.tight_layout()
    return fig

# -----------------------------
# UI Header
# -----------------------------
st.title("🛡️ PaySim Fraud Risk Console")
st.caption("MSIS 522 HW1 — End-to-end fraud detection workflow: EDA → Modeling → SHAP → Streamlit deployment")

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction"
])

# -----------------------------
# Tab 1 — Executive Summary
# -----------------------------
with tab1:
    st.subheader("Problem & Why It Matters")
    st.write(
        """
        This project predicts whether a digital payment transaction is fraudulent using the PaySim dataset
        (simulated mobile money transactions). Fraud is extremely rare (~0.08% in our sample), so we focus
        on metrics that matter for imbalanced classification (F1, PR-AUC), not just accuracy.
        
        **Business goal:** Catch high-risk transactions early to reduce losses, while minimizing false alarms
        that create customer friction and overwhelm fraud review teams.
        """
    )

    st.subheader("Approach")
    st.write(
        """
        1) Performed descriptive analytics to understand fraud patterns (e.g., higher fraud concentration in TRANSFER transactions).  
        2) Trained and compared multiple models: Logistic Regression, Decision Tree, Random Forest, XGBoost, and a Neural Network (MLP).  
        3) Selected the best-performing tree model (XGBoost) and explained predictions using SHAP.  
        4) Deployed an interactive console where users can simulate transactions and view predicted fraud probability and explanations.
        """
    )

    if results_df is not None:
        best_row = results_df.sort_values("F1", ascending=False).iloc[0]
        st.metric("Best Model (by F1)", best_row["Model"])
        st.metric("Best F1", f'{best_row["F1"]:.3f}')
        st.metric("Best PR-AUC", f'{best_row["PR_AUC"]:.3f}')

# -----------------------------
# Tab 2 — Descriptive Analytics
# -----------------------------
with tab2:
    st.subheader("Key Descriptive Insights (from Part 1)")
    st.write(
        """
        In the exploratory analysis, we examined:
        - Class imbalance in fraud labels
        - Fraud rate by transaction type (TRANSFER and CASH_OUT tend to be higher-risk)
        - Amount distributions and balance delta behavior
        - Correlations among numeric features
        
        *(Note: For deployment, we summarize visuals. In your notebook you have the full EDA.)*
        """
    )

    # If you saved EDA plots as images, you can show them here.
    # For now, we show SHAP images only (you can also add EDA images similarly).
    st.info("Tip: If you exported EDA plots as PNGs, drop them into /models and display them here with st.image().")

# -----------------------------
# Tab 3 — Model Performance
# -----------------------------
with tab3:
    st.subheader("Model Comparison")
    if results_df is None:
        st.warning("model_comparison.csv not found in /models. Add it to display model results.")
    else:
        st.dataframe(results_df)

        # Bar chart for F1 (required comparison chart)
        fig = plt.figure()
        plot_df = results_df.sort_values("F1", ascending=False)
        plt.bar(plot_df["Model"], plot_df["F1"])
        plt.title("Model Comparison (F1 on Test Set)")
        plt.ylabel("F1")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Best Hyperparameters")
    if best_params is None:
        st.warning("best_params.json not found in /models. Add it to display tuned hyperparameters.")
    else:
        st.json(best_params)

    st.info(
        "HW note: You should include ROC curves for each model in your app. "
        "If you saved ROC plots from your notebook (as PNGs), place them in /models "
        "and show them here using st.image()."
    )

# -----------------------------
# Tab 4 — Explainability & Interactive Prediction
# -----------------------------
with tab4:
    st.subheader("SHAP Explainability (Best Tree Model)")
    colA, colB = st.columns(2)

    with colA:
        summary_path = os.path.join(MODELS_DIR, "shap_summary.png")
        if os.path.exists(summary_path):
            st.image(summary_path, caption="SHAP Summary (Beeswarm)")
        else:
            st.warning("Missing: models/shap_summary.png")

        bar_path = os.path.join(MODELS_DIR, "shap_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, caption="SHAP Feature Importance (Bar)")
        else:
            st.warning("Missing: models/shap_bar.png")

    with colB:
        waterfall_path = os.path.join(MODELS_DIR, "shap_waterfall.png")
        if os.path.exists(waterfall_path):
            st.image(waterfall_path, caption="SHAP Waterfall (Example Transaction)")
        else:
            st.warning("Missing: models/shap_waterfall.png")

        st.write(
            """
            **How to read SHAP:**  
            - Features with larger SHAP magnitude have stronger impact on the prediction.  
            - Positive SHAP values push the prediction toward fraud; negative values push toward non-fraud.  
            - This makes the model more transparent for analysts and supports audit/compliance needs.
            """
        )

    st.divider()
    st.subheader("Interactive Prediction")

    # Input controls
    c1, c2, c3 = st.columns(3)

    with c1:
        step = st.slider("step (time index)", min_value=1, max_value=744, value=10)
        tx_type = st.selectbox("Transaction type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"], index=4)
        amount = st.number_input("amount", min_value=0.0, value=1000.0, step=50.0)

    with c2:
        old_org = st.number_input("oldbalanceOrg", min_value=0.0, value=5000.0, step=100.0)
        new_org = st.number_input("newbalanceOrig", min_value=0.0, value=4000.0, step=100.0)

    with c3:
        old_dest = st.number_input("oldbalanceDest", min_value=0.0, value=2000.0, step=100.0)
        new_dest = st.number_input("newbalanceDest", min_value=0.0, value=3000.0, step=100.0)

    model_choice = st.selectbox(
        "Choose model",
        ["XGBoost (CV)", "Random Forest (CV)", "Decision Tree (CV)", "Logistic Regression", "MLP (Keras)"],
        index=0
    )

    X_row = build_input_df(step, tx_type, amount, old_org, new_org, old_dest, new_dest)

    if st.button("Predict Fraud Risk"):
        pred_label, pred_prob = predict_with_model(model_choice, X_row)

        st.metric("Predicted Fraud Probability", f"{pred_prob:.4f}")
        st.write("Predicted Class:", "🚨 Fraud" if pred_label == 1 else "✅ Not Fraud")

        st.subheader("SHAP Waterfall for Your Input (XGBoost)")
        st.caption("For consistency, we explain your input using the best tree model (XGBoost).")
        fig = shap_waterfall_for_input(X_row, max_display=15)
        st.pyplot(fig)