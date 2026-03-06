import os
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
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            color: #1f2937;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-bottom: 1.2rem;
        }
        .card {
            background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
            border: 1px solid #e5e7eb;
            padding: 1rem 1.1rem;
            border-radius: 16px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
            margin-bottom: 0.75rem;
        }
        .card h4 {
            margin: 0 0 0.35rem 0;
            font-size: 0.95rem;
            color: #374151;
        }
        .card p {
            margin: 0;
            color: #111827;
            font-size: 1rem;
            font-weight: 600;
        }
        div[data-testid="stMetric"] {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            padding: 0.75rem;
            border-radius: 14px;
        }
        .section-note {
            color: #6b7280;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Utilities
# -----------------------------
def safe_path(*parts) -> str:
    return os.path.join(*parts)

def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)

@st.cache_resource
def load_assets():
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

    for key, fname in [
        ("lr", "lr.pkl"),
        ("tree", "tree.pkl"),
        ("rf", "rf.pkl"),
        ("xgb", "xgb.pkl"),
    ]:
        p = safe_path(MODELS_DIR, fname)
        if file_exists(p):
            assets[key] = joblib.load(p)

    mlp_path = safe_path(MODELS_DIR, "mlp.keras")
    if TF_AVAILABLE and file_exists(mlp_path):
        try:
            assets["mlp"] = keras.models.load_model(mlp_path)  # type: ignore
        except Exception:
            assets["mlp"] = None

    return assets

def make_input_df(
    tx_type: str,
    amount: float,
    oldbalanceOrg: float,
    newbalanceOrig: float,
    oldbalanceDest: float,
    newbalanceDest: float,
    step: int = 1,
) -> pd.DataFrame:
    orig_balance_delta = float(oldbalanceOrg) - float(newbalanceOrig)
    dest_balance_delta = float(newbalanceDest) - float(oldbalanceDest)

    row = {
        "step": int(step),
        "type": tx_type,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest),
        "orig_balance_delta": float(orig_balance_delta),
        "dest_balance_delta": float(dest_balance_delta),
    }
    return pd.DataFrame([row])

def preprocess_row(preprocess, X_df: pd.DataFrame):
    return preprocess.transform(X_df)

def predict_proba(model, Xp):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(Xp)[:, 1][0])

    if hasattr(model, "predict"):
        pred = model.predict(Xp)
        if np.isscalar(pred[0]) and 0.0 <= float(pred[0]) <= 1.0:
            return float(pred[0])

    raise RuntimeError("Model does not support probability prediction.")

def predict_with_selected(model_name: str, assets, X_df: pd.DataFrame):
    # Classical sklearn/xgboost models appear saved as full pipelines
    if model_name == "Logistic Regression":
        if assets["lr"] is None:
            raise FileNotFoundError("lr.pkl not found in /models.")
        return predict_proba(assets["lr"], X_df)

    if model_name == "Decision Tree (CV)":
        if assets["tree"] is None:
            raise FileNotFoundError("tree.pkl not found in /models.")
        return predict_proba(assets["tree"], X_df)

    if model_name == "Random Forest (CV)":
        if assets["rf"] is None:
            raise FileNotFoundError("rf.pkl not found in /models.")
        return predict_proba(assets["rf"], X_df)

    if model_name == "XGBoost (CV)":
        if assets["xgb"] is None:
            raise FileNotFoundError("xgb.pkl not found in /models.")
        return predict_proba(assets["xgb"], X_df)

    if model_name == "MLP (Keras)":
        if (not TF_AVAILABLE) or (assets["mlp"] is None):
            raise RuntimeError("MLP is not available in this deployment (TensorFlow not installed).")

        preprocess = assets["preprocess"]
        Xp = preprocess_row(preprocess, X_df)

        if hasattr(Xp, "toarray"):
            X_dense = Xp.toarray()
        else:
            X_dense = np.array(Xp)

        return float(assets["mlp"].predict(X_dense, verbose=0).ravel()[0])  # type: ignore

    raise ValueError(f"Unknown model: {model_name}")

def risk_label(prob: float, threshold: float = 0.5):
    return "FRAUD" if prob >= threshold else "LEGIT"

def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"

# -----------------------------
# Load models
# -----------------------------
try:
    assets = load_assets()
except Exception as e:
    st.error(f"Failed to load models/preprocess: {e}")
    st.stop()

# -----------------------------
# Data for performance section
# -----------------------------
metrics = [
    {"Model": "XGBoost (CV)",        "Accuracy": 0.999825, "Precision": 0.991304, "Recall": 0.850746, "F1": 0.915663, "ROC_AUC": 0.997263, "PR_AUC": 0.926423},
    {"Model": "Random Forest (CV)",  "Accuracy": 0.999687, "Precision": 0.989848, "Recall": 0.727612, "F1": 0.838710, "ROC_AUC": 0.984813, "PR_AUC": 0.916957},
    {"Model": "MLP (Keras)",         "Accuracy": 0.999487, "Precision": 0.993197, "Recall": 0.544776, "F1": 0.703614, "ROC_AUC": 0.979838, "PR_AUC": 0.741544},
    {"Model": "Decision Tree (CV)",  "Accuracy": 0.991592, "Precision": 0.110419, "Recall": 0.925373, "F1": 0.197295, "ROC_AUC": 0.961404, "PR_AUC": 0.622175},
    {"Model": "Logistic Regression", "Accuracy": 0.940196, "Precision": 0.017868, "Recall": 0.973881, "F1": 0.035092, "ROC_AUC": 0.984469, "PR_AUC": 0.584643},
]
df_metrics = pd.DataFrame(metrics)
best_row = df_metrics.sort_values(["F1", "ROC_AUC"], ascending=False).iloc[0]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🛡️ Fraud Risk Console")
st.sidebar.caption("PaySim-based fraud scoring demo")

threshold = st.sidebar.slider("Fraud decision threshold", 0.05, 0.95, 0.50, 0.01)

st.sidebar.divider()

model_options = [
    "XGBoost (CV)",
    "Random Forest (CV)",
    "Decision Tree (CV)",
    "Logistic Regression",
]
if TF_AVAILABLE and assets.get("mlp") is not None:
    model_options.append("MLP (Keras)")

selected_model = st.sidebar.selectbox("Choose model", model_options, index=0)

if not TF_AVAILABLE:
    st.sidebar.info("TensorFlow unavailable in this environment, so MLP is hidden automatically.")

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">PaySim Fraud Risk Console</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive fraud scoring, model comparison, and explainability for PaySim transaction data.</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Best Model", best_row["Model"])
with m2:
    st.metric("Best F1 Score", f'{best_row["F1"]:.3f}')
with m3:
    st.metric("Best ROC-AUC", f'{best_row["ROC_AUC"]:.3f}')

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Summary", "Descriptive Analytics", "Model Performance", "Explainability & Prediction"]
)

# -----------------------------
# Tab 1 - Executive Summary
# -----------------------------
with tab1:
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown(
            """
            <div class="card">
                <h4>Business Problem</h4>
                <p>Detect high-risk financial transactions before funds leave the system.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="card">
                <h4>Dataset</h4>
                <p>PaySim synthetic mobile money transactions with fraud labels.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="card">
                <h4>Approach</h4>
                <p>Preprocessing + supervised classification + threshold-based decisioning + SHAP explainability.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown("### Key takeaway")
        st.write(
            f"""
            Among the evaluated models, **{best_row["Model"]}** performed best on the fraud detection task.
            The dashboard supports interactive transaction scoring, threshold tuning, side-by-side model review,
            and explainability artifacts to support decision transparency.
            """
        )

        st.markdown("### Why this matters")
        st.write(
            """
            Fraud detection is a high-imbalance classification problem where overall accuracy can be misleading.
            For this use case, recall, F1, PR-AUC, and ROC-AUC are more useful for evaluating how effectively
            the model identifies fraudulent transactions while controlling false positives.
            """
        )

# -----------------------------
# Tab 2 - Descriptive Analytics
# -----------------------------
with tab2:
    st.markdown("### Descriptive Analytics")
    st.markdown(
        '<div class="section-note">Add your Part 1 visuals here so the app fully reflects the HW workflow.</div>',
        unsafe_allow_html=True
    )

    eda1 = safe_path(MODELS_DIR, "eda_class_distribution.png")
    eda2 = safe_path(MODELS_DIR, "eda_transaction_type.png")
    eda3 = safe_path(MODELS_DIR, "eda_amount_distribution.png")
    eda4 = safe_path(MODELS_DIR, "eda_correlation_heatmap.png")

    c1, c2 = st.columns(2)

    with c1:
        if file_exists(eda1):
            st.image(eda1, caption="Fraud vs Non-Fraud Distribution", use_container_width=True)
        else:
            st.info("Add `models/eda_class_distribution.png` for fraud class distribution.")

        if file_exists(eda3):
            st.image(eda3, caption="Transaction Amount Distribution", use_container_width=True)
        else:
            st.info("Add `models/eda_amount_distribution.png` for amount distribution.")

    with c2:
        if file_exists(eda2):
            st.image(eda2, caption="Transaction Type Analysis", use_container_width=True)
        else:
            st.info("Add `models/eda_transaction_type.png` for transaction type analysis.")

        if file_exists(eda4):
            st.image(eda4, caption="Feature Correlation Heatmap", use_container_width=True)
        else:
            st.info("Add `models/eda_correlation_heatmap.png` for correlation heatmap.")

    st.markdown("### Interpretation")
    st.write(
        """
        Use this section to summarize the most important descriptive findings:
        which transaction types dominate fraud, how imbalanced the target is,
        and which variables appear most informative before modeling.
        """
    )

# -----------------------------
# Tab 3 - Model Performance
# -----------------------------
with tab3:
    st.markdown("### Model Performance Summary")

    if not (TF_AVAILABLE and assets.get("mlp") is not None):
        st.info("MLP metrics are shown from training results, though the deployed app may disable MLP scoring if TensorFlow is unavailable.")

    display_df = df_metrics.copy()
    st.dataframe(
        display_df.style.highlight_max(subset=["F1", "ROC_AUC", "PR_AUC"], color="#dbeafe"),
        use_container_width=True
    )

    st.markdown("### Metric Comparison")
    chart_df = display_df.set_index("Model")[["F1", "ROC_AUC", "PR_AUC"]]
    st.bar_chart(chart_df)

    st.markdown("### Best Hyperparameters")
    best_params_path = safe_path(MODELS_DIR, "best_params.json")
    if file_exists(best_params_path):
        try:
            with open(best_params_path, "r", encoding="utf-8") as f:
                best_params = f.read()
            st.code(best_params, language="json")
        except Exception as e:
            st.warning(f"Could not load best_params.json: {e}")
    else:
        st.info("Add `models/best_params.json` if you want tuned hyperparameters displayed here.")

    st.markdown("### ROC / Precision-Recall Curves")

    roc_lr = safe_path(MODELS_DIR, "logistic_regression_roc_curve.png")
    pr_lr = safe_path(MODELS_DIR, "logistic_regression_pr_curve.png")
    roc_rf = safe_path(MODELS_DIR, "random_forest_roc_curve.png")
    pr_rf = safe_path(MODELS_DIR, "random_forest_pr_curve.png")
    roc_dt = safe_path(MODELS_DIR, "decision_tree_roc_curve.png")
    pr_dt = safe_path(MODELS_DIR, "decision_tree_pr_curve.png")

    c1, c2 = st.columns(2)

    with c1:
        if file_exists(roc_rf):
            st.image(roc_rf, caption="Random Forest ROC Curve", use_container_width=True)
        else:
            st.info("Add models/random_forest_roc_curve.png")

        if file_exists(roc_lr):
            st.image(roc_lr, caption="Logistic Regression ROC Curve", use_container_width=True)
        else:
            st.info("Add models/logistic_regression_roc_curve.png")

        if file_exists(roc_dt):
            st.image(roc_dt, caption="Decision Tree ROC Curve", use_container_width=True)
        else:
            st.info("Add models/decision_tree_roc_curve.png")

    with c2:
        if file_exists(pr_rf):
            st.image(pr_rf, caption="Random Forest Precision-Recall Curve", use_container_width=True)
        else:
            st.info("Add models/random_forest_pr_curve.png")

        if file_exists(pr_lr):
            st.image(pr_lr, caption="Logistic Regression Precision-Recall Curve", use_container_width=True)
        else:
            st.info("Add models/logistic_regression_pr_curve.png")

        if file_exists(pr_dt):
            st.image(pr_dt, caption="Decision Tree Precision-Recall Curve", use_container_width=True)
        else:
            st.info("Add models/decision_tree_pr_curve.png")

# -----------------------------
# Tab 4 - Explainability & Prediction
# -----------------------------
with tab4:
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### Interactive Prediction")

        with st.form("tx_form", clear_on_submit=False):
            step = st.number_input("Step (time step)", min_value=1, value=1, step=1)

            tx_type = st.selectbox(
                "Transaction type",
                ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
            )

            amount = st.number_input("Amount", min_value=0.0, value=100000.0, step=100.0)

            oldbalanceOrg = st.number_input(
                "Origin old balance (oldbalanceOrg)",
                min_value=0.0,
                value=100000.0,
                step=100.0
            )

            newbalanceOrig = st.number_input(
                "Origin new balance (newbalanceOrig)",
                min_value=0.0,
                value=0.0,
                step=100.0
            )

            oldbalanceDest = st.number_input(
                "Destination old balance (oldbalanceDest)",
                min_value=0.0,
                value=0.0,
                step=100.0
            )

            newbalanceDest = st.number_input(
                "Destination new balance (newbalanceDest)",
                min_value=0.0,
                value=0.0,
                step=100.0
            )

            submitted = st.form_submit_button("Score transaction")

        if submitted:
            X_df = make_input_df(
                step=step,
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

                r1, r2 = st.columns(2)
                with r1:
                    st.metric("Fraud Probability", format_pct(prob))
                with r2:
                    st.metric("Decision", decision)

                if decision == "FRAUD":
                    st.error("🚨 High-risk transaction flagged as FRAUD.")
                else:
                    st.success("✅ Transaction classified as LEGIT.")

                with st.expander("Show model input row"):
                    st.dataframe(X_df, use_container_width=True)

            except Exception as e:
                st.error(f"Scoring failed: {e}")

    with right:
        st.markdown("### Explainability Artifacts (SHAP)")
        st.write(f"**Model selected:** {selected_model}")
        st.write(f"**Decision threshold:** {threshold:.2f}")

        shap_summary = safe_path(MODELS_DIR, "shap_summary.png")
        shap_bar = safe_path(MODELS_DIR, "shap_bar.png")
        shap_waterfall = safe_path(MODELS_DIR, "shap_waterfall.png")

        if file_exists(shap_summary):
            st.image(shap_summary, caption="SHAP Summary Plot", use_container_width=True)
        else:
            st.warning("Missing models/shap_summary.png")

        if file_exists(shap_bar):
            st.image(shap_bar, caption="SHAP Feature Importance (Bar)", use_container_width=True)
        else:
            st.warning("Missing models/shap_bar.png")

        if file_exists(shap_waterfall):
            st.image(shap_waterfall, caption="SHAP Waterfall (Example Prediction)", use_container_width=True)
        else:
            st.caption("Optional: add models/shap_waterfall.png for a single-prediction explanation.")

