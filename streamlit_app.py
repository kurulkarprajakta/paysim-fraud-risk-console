import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Optional TF (so app doesn't crash on Streamlit Cloud)
try:
    from tensorflow import keras  # type: ignore
    TF_AVAILABLE = True
except Exception:
    keras = None
    TF_AVAILABLE = False

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="PaySim Fraud Risk Console",
    page_icon="🛡️",
    layout="wide",
)

MODELS_DIR = "models"

# -----------------------------
# Utilities
# -----------------------------
def safe_path(*parts) -> str:
    return os.path.join(*parts)

def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)

@st.cache_resource
def load_assets():
    """Load saved models + preprocess. MLP is optional."""
    preprocess_path = safe_path(MODELS_DIR, "preprocess.pkl")
    if not file_exists(preprocess_path):
        raise FileNotFoundError(
            f"Missing {preprocess_path}. Ensure preprocess.pkl exists in /models."
        )
    preprocess = joblib.load(preprocess_path)

    assets = {
        "preprocess": preprocess,
        "lr": None,
        "tree": None,
        "rf": None,
        "xgb": None,
        "mlp": None,
    }

    # Load classical ML models if present
    for key, fname in [
        ("lr", "lr.pkl"),
        ("tree", "tree.pkl"),
        ("rf", "rf.pkl"),
        ("xgb", "xgb.pkl"),
    ]:
        p = safe_path(MODELS_DIR, fname)
        if file_exists(p):
            assets[key] = joblib.load(p)

    # Load MLP only if TF available + file exists
    mlp_path = safe_path(MODELS_DIR, "mlp.keras")
    if TF_AVAILABLE and file_exists(mlp_path):
        try:
            assets["mlp"] = keras.models.load_model(mlp_path)  # type: ignore
        except Exception:
            assets["mlp"] = None

    return assets

def make_input_df(tx_type: str,
                  amount: float,
                  oldbalanceOrg: float,
                  newbalanceOrig: float,
                  oldbalanceDest: float,
                  newbalanceDest: float) -> pd.DataFrame:
    """
    Create a single-row dataframe matching typical PaySim features.
    Adjust column names here if your notebook used different names.
    """
    row = {
        "type": tx_type,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest),
    }
    return pd.DataFrame([row])

def preprocess_row(preprocess, X_df: pd.DataFrame):
    """Transform input row with saved preprocess pipeline."""
    Xp = preprocess.transform(X_df)
    return Xp

def predict_proba(model, Xp):
    """Return fraud probability for sklearn / xgboost models."""
    # Some models have predict_proba
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(Xp)[:, 1][0])
        return proba
    # Some xgboost versions may not expose predict_proba (rare)
    if hasattr(model, "predict"):
        pred = model.predict(Xp)
        # If it returns a probability already:
        if np.isscalar(pred[0]) and 0.0 <= float(pred[0]) <= 1.0:
            return float(pred[0])
    raise RuntimeError("Model does not support probability prediction.")

def predict_with_selected(model_name: str, assets, X_df: pd.DataFrame):
    preprocess = assets["preprocess"]
    Xp = preprocess_row(preprocess, X_df)

    if model_name == "Logistic Regression":
        if assets["lr"] is None:
            raise FileNotFoundError("lr.pkl not found in /models.")
        proba = predict_proba(assets["lr"], Xp)
        return proba

    if model_name == "Decision Tree (CV)":
        if assets["tree"] is None:
            raise FileNotFoundError("tree.pkl not found in /models.")
        proba = predict_proba(assets["tree"], Xp)
        return proba

    if model_name == "Random Forest (CV)":
        if assets["rf"] is None:
            raise FileNotFoundError("rf.pkl not found in /models.")
        proba = predict_proba(assets["rf"], Xp)
        return proba

    if model_name == "XGBoost (CV)":
        if assets["xgb"] is None:
            raise FileNotFoundError("xgb.pkl not found in /models.")
        proba = predict_proba(assets["xgb"], Xp)
        return proba

    if model_name == "MLP (Keras)":
        # MLP is optional; fail gracefully
        if (not TF_AVAILABLE) or (assets["mlp"] is None):
            raise RuntimeError("MLP is not available in this deployment (TensorFlow not installed).")

        # Keras needs dense
        if hasattr(Xp, "toarray"):
            X_dense = Xp.toarray()
        else:
            X_dense = np.array(Xp)

        prob = float(assets["mlp"].predict(X_dense, verbose=0).ravel()[0])  # type: ignore
        return prob

    raise ValueError(f"Unknown model: {model_name}")

def risk_label(prob: float, threshold: float = 0.5):
    label = "FRAUD" if prob >= threshold else "LEGIT"
    return label

def format_pct(x: float) -> str:
    return f"{x*100:.2f}%"

# -----------------------------
# Load models
# -----------------------------
try:
    assets = load_assets()
except Exception as e:
    st.error(f"Failed to load models/preprocess: {e}")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🛡️ Fraud Risk Console")
st.sidebar.caption("PaySim-based fraud scoring demo")

threshold = st.sidebar.slider("Fraud decision threshold", 0.05, 0.95, 0.50, 0.01)

st.sidebar.divider()

# Model options (MLP only if available)
model_options = ["XGBoost (CV)", "Random Forest (CV)", "Decision Tree (CV)", "Logistic Regression"]
if TF_AVAILABLE and assets.get("mlp") is not None:
    model_options.append("MLP (Keras)")

selected_model = st.sidebar.selectbox("Choose model", model_options, index=0)

if not TF_AVAILABLE:
    st.sidebar.info("TensorFlow not available on this environment → MLP disabled automatically.")

# -----------------------------
# Main layout
# -----------------------------
st.title("PaySim Fraud Risk Console")
st.caption("Interactive scoring + model comparison + explainability artifacts (SHAP).")

colA, colB = st.columns([1.2, 1])

with colA:
    st.subheader("1) Enter transaction details")
    with st.form("tx_form", clear_on_submit=False):
        tx_type = st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
        oldbalanceOrg = st.number_input("Origin old balance (oldbalanceOrg)", min_value=0.0, value=5000.0, step=100.0)
        newbalanceOrig = st.number_input("Origin new balance (newbalanceOrig)", min_value=0.0, value=4000.0, step=100.0)
        oldbalanceDest = st.number_input("Destination old balance (oldbalanceDest)", min_value=0.0, value=0.0, step=100.0)
        newbalanceDest = st.number_input("Destination new balance (newbalanceDest)", min_value=0.0, value=0.0, step=100.0)

        submitted = st.form_submit_button("Score transaction")

with colB:
    st.subheader("2) Score & decision")
    st.write(f"**Model selected:** {selected_model}")
    st.write(f"**Threshold:** {threshold:.2f}")

    if submitted:
        X_df = make_input_df(
            tx_type=tx_type,
            amount=amount,
            oldbalanceOrg=oldbalanceOrg,
            newbalanceOrig=newbalanceOrig,
            oldbalanceDest=oldbalanceDest,
            newbalanceDest=newbalanceDest,
        )

        try:
            prob = predict_with_selected(selected_model, assets, X_df)
            decision = risk_label(prob, threshold=threshold)

            if decision == "FRAUD":
                st.error(f"🚨 Decision: **{decision}**")
            else:
                st.success(f"✅ Decision: **{decision}**")

            st.metric("Fraud probability", format_pct(prob))

            with st.expander("Show model input row"):
                st.dataframe(X_df, use_container_width=True)

        except Exception as e:
            st.error(f"Scoring failed: {e}")

st.divider()

# -----------------------------
# Model comparison table (your results)
# -----------------------------
st.subheader("3) Model performance summary (from your HW run)")

metrics = [
    {"Model": "XGBoost (CV)",        "Accuracy": 0.999825, "Precision": 0.991304, "Recall": 0.850746, "F1": 0.915663, "ROC_AUC": 0.997263, "PR_AUC": 0.926423},
    {"Model": "Random Forest (CV)",  "Accuracy": 0.999687, "Precision": 0.989848, "Recall": 0.727612, "F1": 0.838710, "ROC_AUC": 0.984813, "PR_AUC": 0.916957},
    {"Model": "MLP (Keras)",         "Accuracy": 0.999487, "Precision": 0.993197, "Recall": 0.544776, "F1": 0.703614, "ROC_AUC": 0.979838, "PR_AUC": 0.741544},
    {"Model": "Decision Tree (CV)",  "Accuracy": 0.991592, "Precision": 0.110419, "Recall": 0.925373, "F1": 0.197295, "ROC_AUC": 0.961404, "PR_AUC": 0.622175},
    {"Model": "Logistic Regression", "Accuracy": 0.940196, "Precision": 0.017868, "Recall": 0.973881, "F1": 0.035092, "ROC_AUC": 0.984469, "PR_AUC": 0.584643},
]
df_metrics = pd.DataFrame(metrics)

# Note about MLP availability
if not (TF_AVAILABLE and assets.get("mlp") is not None):
    st.info("Note: MLP scores above are from training results, but the MLP model may be disabled in this deployment if TensorFlow is unavailable.")

st.dataframe(df_metrics, use_container_width=True)

st.divider()

# -----------------------------
# Explainability / artifacts
# -----------------------------
st.subheader("4) Explainability artifacts (SHAP)")

shap_summary = safe_path(MODELS_DIR, "shap_summary.png")
shap_bar = safe_path(MODELS_DIR, "shap_bar.png")
shap_waterfall = safe_path(MODELS_DIR, "shap_waterfall.png")

c1, c2 = st.columns(2)
with c1:
    if file_exists(shap_summary):
        st.image(shap_summary, caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.warning("Missing models/shap_summary.png")

with c2:
    if file_exists(shap_bar):
        st.image(shap_bar, caption="SHAP Feature Importance (Bar)", use_container_width=True)
    else:
        st.warning("Missing models/shap_bar.png")

if file_exists(shap_waterfall):
    st.image(shap_waterfall, caption="SHAP Waterfall (Example)", use_container_width=True)
else:
    st.caption("Optional: add models/shap_waterfall.png for a single-prediction explanation.")

st.divider()

# -----------------------------
# Footer / HW submission help
# -----------------------------
st.subheader("5) What to submit (HW checklist)")
st.markdown(
    """
- ✅ **GitHub repo link** (this repo)
- ✅ **Streamlit app link** (this deployment URL)
- ✅ Brief explanation: problem, dataset (PaySim), preprocessing, models tried, evaluation metrics
- ✅ Explainability: SHAP summary + bar plots (screenshots/embedded)
"""
)