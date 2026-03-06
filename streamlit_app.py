import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Optional TF
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
            padding-top: 0.8rem;
            padding-bottom: 1.5rem;
            max-width: 1320px;
        }
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #1d4ed8 100%);
            color: white;
            padding: 1.35rem 1.5rem;
            border-radius: 22px;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        .hero p {
            margin: 0.45rem 0 0 0;
            color: rgba(255,255,255,0.86);
            font-size: 1rem;
        }
        .soft-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: 0 2px 12px rgba(15,23,42,0.05);
        }
        .small-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.04);
        }
        .small-label {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .small-value {
            color: #0f172a;
            font-size: 1.2rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .section-title {
            font-size: 1.55rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.2rem;
        }
        .section-sub {
            color: #6b7280;
            margin-bottom: 0.8rem;
        }
        .risk-high {
            background: #fef2f2;
            color: #b91c1c;
            border: 1px solid #fecaca;
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            display: inline-block;
            font-weight: 700;
            font-size: 0.9rem;
        }
        .risk-low {
            background: #ecfdf5;
            color: #047857;
            border: 1px solid #a7f3d0;
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            display: inline-block;
            font-weight: 700;
            font-size: 0.9rem;
        }
        .insight {
            background: #f8fafc;
            border-left: 4px solid #2563eb;
            padding: 0.85rem 1rem;
            border-radius: 12px;
            color: #334155;
            margin-bottom: 0.65rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 0.6rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.04);
        }
        div[data-testid="stTabs"] button {
            font-weight: 700;
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
        raise FileNotFoundError(f"Missing {preprocess_path}")
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
    if model_name == "Logistic Regression":
        if assets["lr"] is None:
            raise FileNotFoundError("lr.pkl not found.")
        return predict_proba(assets["lr"], X_df)

    if model_name == "Decision Tree (CV)":
        if assets["tree"] is None:
            raise FileNotFoundError("tree.pkl not found.")
        return predict_proba(assets["tree"], X_df)

    if model_name == "Random Forest (CV)":
        if assets["rf"] is None:
            raise FileNotFoundError("rf.pkl not found.")
        return predict_proba(assets["rf"], X_df)

    if model_name == "XGBoost (CV)":
        if assets["xgb"] is None:
            raise FileNotFoundError("xgb.pkl not found.")
        return predict_proba(assets["xgb"], X_df)

    if model_name == "MLP (Keras)":
        if (not TF_AVAILABLE) or (assets["mlp"] is None):
            raise RuntimeError("MLP is unavailable in this deployment.")
        preprocess = assets["preprocess"]
        Xp = preprocess_row(preprocess, X_df)
        X_dense = Xp.toarray() if hasattr(Xp, "toarray") else np.array(Xp)
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
# Metrics
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

model_options = ["XGBoost (CV)", "Random Forest (CV)", "Decision Tree (CV)", "Logistic Regression"]
if TF_AVAILABLE and assets.get("mlp") is not None:
    model_options.append("MLP (Keras)")

selected_model = st.sidebar.selectbox("Choose model", model_options, index=0)

scenario = st.sidebar.selectbox(
    "Quick scenario",
    [
        "Custom",
        "High-risk transfer",
        "Medium-risk cash-out",
        "Low-risk payment",
    ],
)

if scenario == "High-risk transfer":
    default_step = 1
    default_type = "TRANSFER"
    default_amount = 100000.0
    default_old_org = 100000.0
    default_new_org = 0.0
    default_old_dest = 0.0
    default_new_dest = 0.0
elif scenario == "Medium-risk cash-out":
    default_step = 20
    default_type = "CASH_OUT"
    default_amount = 20000.0
    default_old_org = 25000.0
    default_new_org = 3000.0
    default_old_dest = 5000.0
    default_new_dest = 25000.0
elif scenario == "Low-risk payment":
    default_step = 10
    default_type = "PAYMENT"
    default_amount = 120.0
    default_old_org = 2000.0
    default_new_org = 1880.0
    default_old_dest = 0.0
    default_new_dest = 0.0
else:
    default_step = 1
    default_type = "TRANSFER"
    default_amount = 100000.0
    default_old_org = 100000.0
    default_new_org = 0.0
    default_old_dest = 0.0
    default_new_dest = 0.0

st.sidebar.divider()
st.sidebar.markdown(
    f"""
**Active model:** {selected_model}  
**Threshold:** {threshold:.2f}
"""
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>PaySim Fraud Risk Console</h1>
        <p>Interactive fraud detection dashboard combining exploratory analysis, model benchmarking, explainability, and live transaction scoring.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="small-card"><div class="small-label">Best Model</div><div class="small-value">{best_row["Model"]}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="small-card"><div class="small-label">Best F1</div><div class="small-value">{best_row["F1"]:.3f}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="small-card"><div class="small-label">ROC-AUC</div><div class="small-value">{best_row["ROC_AUC"]:.3f}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="small-card"><div class="small-label">PR-AUC</div><div class="small-value">{best_row["PR_AUC"]:.3f}</div></div>', unsafe_allow_html=True)

st.write("")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Summary", "Descriptive Analytics", "Model Performance", "Prediction & Explainability"]
)

# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    a, b = st.columns([1.15, 1])

    with a:
        st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Business context, modeling approach, and deployment outcome.</div>', unsafe_allow_html=True)

        st.markdown('<div class="insight"><b>Problem:</b> Detect fraudulent mobile money transactions in a highly imbalanced setting where false negatives are costly.</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight"><b>Dataset:</b> PaySim transaction data with engineered balance-delta features and categorical transaction types.</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight"><b>Approach:</b> Preprocessing + multiple classifiers + tuned decision thresholds + SHAP for interpretability.</div>', unsafe_allow_html=True)

    with b:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("#### Key takeaway")
        st.write(
            f"""
            **{best_row["Model"]}** emerged as the strongest overall model in this workflow,
            delivering the best balance of **F1**, **ROC-AUC**, and **PR-AUC** for fraud detection.
            The deployed console supports live risk scoring, model comparison, and explanation artifacts.
            """
        )
        st.markdown("#### Why this matters")
        st.write(
            """
            Fraud detection should not be judged by accuracy alone. Because fraud is rare, metrics like
            **Recall**, **F1**, and especially **PR-AUC** provide a more realistic measure of model usefulness.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Tab 2
# -----------------------------
with tab2:
    st.markdown('<div class="section-title">Descriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Key exploratory findings from the transaction dataset.</div>', unsafe_allow_html=True)

    eda1 = safe_path(MODELS_DIR, "eda_class_distribution.png")
    eda2 = safe_path(MODELS_DIR, "eda_transaction_type.png")
    eda3 = safe_path(MODELS_DIR, "eda_amount_distribution.png")
    eda4 = safe_path(MODELS_DIR, "eda_correlation_heatmap.png")

    c1, c2 = st.columns(2)
    with c1:
        if file_exists(eda1):
            st.image(eda1, caption="Target Distribution: extreme fraud imbalance", use_container_width=True)
        if file_exists(eda3):
            st.image(eda3, caption="Log-scaled amount distribution by fraud label", use_container_width=True)

    with c2:
        if file_exists(eda2):
            st.image(eda2, caption="Fraud rate is concentrated in TRANSFER and CASH_OUT", use_container_width=True)
        if file_exists(eda4):
            st.image(eda4, caption="Correlation structure across numeric features", use_container_width=True)

    st.markdown("#### Key EDA insights")
    st.markdown(
        """
- Fraud is extremely rare, making the task heavily imbalanced.
- Fraud concentrates in **TRANSFER** and **CASH_OUT**, while other transaction types contribute very little.
- Transaction amounts are highly skewed, so log scaling improves interpretability.
- Engineered balance-delta features appear more informative than raw balances alone.
        """
    )

# -----------------------------
# Tab 3
# -----------------------------
with tab3:
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Benchmarking across classifiers, metrics, and diagnostic curves.</div>', unsafe_allow_html=True)

    st.dataframe(
        df_metrics.style.highlight_max(subset=["F1", "ROC_AUC", "PR_AUC"], color="#dbeafe"),
        use_container_width=True,
    )

    st.markdown("#### Metric Comparison")
    chart_df = df_metrics.set_index("Model")[["F1", "ROC_AUC", "PR_AUC"]]
    st.bar_chart(chart_df)

    with st.expander("Best Hyperparameters", expanded=False):
        best_params_path = safe_path(MODELS_DIR, "best_params.json")
        if file_exists(best_params_path):
            with open(best_params_path, "r", encoding="utf-8") as f:
                st.code(f.read(), language="json")
        else:
            st.info("Add models/best_params.json")

    with st.expander("ROC / PR Curve Diagnostics", expanded=False):
        roc_lr = safe_path(MODELS_DIR, "logistic_regression_roc_curve.png")
        pr_lr = safe_path(MODELS_DIR, "logistic_regression_pr_curve.png")
        roc_rf = safe_path(MODELS_DIR, "random_forest_roc_curve.png")
        pr_rf = safe_path(MODELS_DIR, "random_forest_pr_curve.png")
        roc_dt = safe_path(MODELS_DIR, "decision_tree_roc_curve.png")
        pr_dt = safe_path(MODELS_DIR, "decision_tree_pr_curve.png")

        x1, x2 = st.columns(2)
        with x1:
            if file_exists(roc_rf):
                st.image(roc_rf, caption="Random Forest ROC Curve", use_container_width=True)
            if file_exists(roc_lr):
                st.image(roc_lr, caption="Logistic Regression ROC Curve", use_container_width=True)
            if file_exists(roc_dt):
                st.image(roc_dt, caption="Decision Tree ROC Curve", use_container_width=True)
        with x2:
            if file_exists(pr_rf):
                st.image(pr_rf, caption="Random Forest Precision-Recall Curve", use_container_width=True)
            if file_exists(pr_lr):
                st.image(pr_lr, caption="Logistic Regression Precision-Recall Curve", use_container_width=True)
            if file_exists(pr_dt):
                st.image(pr_dt, caption="Decision Tree Precision-Recall Curve", use_container_width=True)

# -----------------------------
# Tab 4
# -----------------------------
with tab4:
    left, right = st.columns([1.02, 1])

    with left:
        st.markdown('<div class="section-title">Interactive Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Score a transaction and assess risk under the selected model and threshold.</div>', unsafe_allow_html=True)

        with st.form("tx_form", clear_on_submit=False):
            step = st.number_input("Step", min_value=1, value=int(default_step), step=1)
            tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                                   index=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"].index(default_type))
            amount = st.number_input("Amount", min_value=0.0, value=float(default_amount), step=100.0)
            oldbalanceOrg = st.number_input("Origin Old Balance", min_value=0.0, value=float(default_old_org), step=100.0)
            newbalanceOrig = st.number_input("Origin New Balance", min_value=0.0, value=float(default_new_org), step=100.0)
            oldbalanceDest = st.number_input("Destination Old Balance", min_value=0.0, value=float(default_old_dest), step=100.0)
            newbalanceDest = st.number_input("Destination New Balance", min_value=0.0, value=float(default_new_dest), step=100.0)
            submitted = st.form_submit_button("Run Risk Check")

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

                st.session_state["last_prob"] = prob
                st.session_state["last_decision"] = decision
                st.session_state["last_input"] = X_df

            except Exception as e:
                st.error(f"Scoring failed: {e}")

        if "last_prob" in st.session_state:
            prob = st.session_state["last_prob"]
            decision = st.session_state["last_decision"]
            X_df = st.session_state["last_input"]

            r1, r2 = st.columns(2)
            with r1:
                st.metric("Fraud Probability", format_pct(prob))
            with r2:
                st.metric("Decision Threshold", f"{threshold:.2f}")

            st.progress(min(max(prob, 0.0), 1.0))

            if decision == "FRAUD":
                st.markdown('<span class="risk-high">High Risk • FRAUD</span>', unsafe_allow_html=True)
                st.error("This transaction is flagged as high risk under the current threshold.")
            else:
                st.markdown('<span class="risk-low">Low Risk • LEGIT</span>', unsafe_allow_html=True)
                st.success("This transaction is classified as legitimate under the current threshold.")

            with st.expander("Show model input row", expanded=False):
                st.dataframe(X_df, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Explainability</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Global SHAP artifacts for interpreting model behavior.</div>', unsafe_allow_html=True)

        info1, info2 = st.columns(2)
        with info1:
            st.markdown(f'<div class="small-card"><div class="small-label">Selected Model</div><div class="small-value">{selected_model}</div></div>', unsafe_allow_html=True)
        with info2:
            st.markdown(f'<div class="small-card"><div class="small-label">Threshold</div><div class="small-value">{threshold:.2f}</div></div>', unsafe_allow_html=True)

        shap_summary = safe_path(MODELS_DIR, "shap_summary.png")
        shap_bar = safe_path(MODELS_DIR, "shap_bar.png")
        shap_waterfall = safe_path(MODELS_DIR, "shap_waterfall.png")

        if file_exists(shap_summary):
            st.image(shap_summary, caption="SHAP Summary Plot", use_container_width=True)
        if file_exists(shap_bar):
            st.image(shap_bar, caption="SHAP Feature Importance", use_container_width=True)

        with st.expander("Single-Prediction Waterfall Example", expanded=False):
            if file_exists(shap_waterfall):
                st.image(shap_waterfall, caption="SHAP Waterfall Plot", use_container_width=True)
            else:
                st.info("Add models/shap_waterfall.png")
