import os
import json
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


# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(
    page_title="PaySim Fraud Risk Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = "models"


# ---------------------------------------------------
# Styling
# ---------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 1.5rem;
            max-width: 1380px;
        }

        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #172554 40%, #1d4ed8 100%);
            color: white;
            padding: 1.4rem 1.6rem;
            border-radius: 24px;
            margin-bottom: 1rem;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18);
        }

        .hero-title {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0;
            letter-spacing: -0.03em;
        }

        .hero-subtitle {
            margin-top: 0.45rem;
            font-size: 1rem;
            color: rgba(255,255,255,0.88);
            max-width: 980px;
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.05);
            margin-bottom: 0.5rem;
        }

        .metric-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-value {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }

        .section-title {
            font-size: 1.55rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.15rem;
        }

        .section-subtitle {
            color: #6b7280;
            margin-bottom: 0.85rem;
        }

        .soft-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: 0 2px 12px rgba(15,23,42,0.05);
        }

        .insight-box {
            background: #f8fafc;
            border-left: 4px solid #2563eb;
            border-radius: 12px;
            padding: 0.8rem 0.95rem;
            color: #334155;
            margin-bottom: 0.7rem;
        }

        .risk-high {
            background: #fef2f2;
            color: #b91c1c;
            border: 1px solid #fecaca;
            border-radius: 999px;
            padding: 0.38rem 0.8rem;
            display: inline-block;
            font-weight: 800;
            font-size: 0.9rem;
        }

        .risk-low {
            background: #ecfdf5;
            color: #047857;
            border: 1px solid #a7f3d0;
            border-radius: 999px;
            padding: 0.38rem 0.8rem;
            display: inline-block;
            font-weight: 800;
            font-size: 0.9rem;
        }

        .risk-review {
            background: #fffbeb;
            color: #b45309;
            border: 1px solid #fde68a;
            border-radius: 999px;
            padding: 0.38rem 0.8rem;
            display: inline-block;
            font-weight: 800;
            font-size: 0.9rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 0.65rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.04);
        }

        div[data-testid="stTabs"] button {
            font-weight: 700;
        }

        .tiny-note {
            color: #6b7280;
            font-size: 0.92rem;
        }

        .recommend-box {
            background: linear-gradient(180deg, #eff6ff 0%, #f8fbff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-top: 0.8rem;
            color: #1e3a8a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
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
        if len(pred) > 0 and np.isscalar(pred[0]) and 0.0 <= float(pred[0]) <= 1.0:
            return float(pred[0])

    raise RuntimeError("Model does not support probability prediction.")


def predict_with_selected(model_name: str, assets, X_df: pd.DataFrame):
    # sklearn/xgboost models are saved as full pipelines
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
            raise RuntimeError("MLP is not available in this deployment.")
        preprocess = assets["preprocess"]
        Xp = preprocess_row(preprocess, X_df)
        X_dense = Xp.toarray() if hasattr(Xp, "toarray") else np.array(Xp)
        return float(assets["mlp"].predict(X_dense, verbose=0).ravel()[0])  # type: ignore

    raise ValueError(f"Unknown model: {model_name}")


def risk_label(prob: float, threshold: float = 0.5):
    return "FRAUD" if prob >= threshold else "LEGIT"


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def recommended_action(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "Escalate for fraud review"
    if prob >= max(0.30, threshold - 0.15):
        return "Send to manual review"
    return "Approve transaction"


def risk_badge(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return '<span class="risk-high">High Risk • FRAUD</span>'
    if prob >= max(0.30, threshold - 0.15):
        return '<span class="risk-review">Moderate Risk • REVIEW</span>'
    return '<span class="risk-low">Low Risk • LEGIT</span>'


# ---------------------------------------------------
# Assets + data
# ---------------------------------------------------
try:
    assets = load_assets()
except Exception as e:
    st.error(f"Failed to load models/preprocess: {e}")
    st.stop()

metrics = [
    {"Model": "XGBoost (CV)",        "Accuracy": 0.999825, "Precision": 0.991304, "Recall": 0.850746, "F1": 0.915663, "ROC_AUC": 0.997263, "PR_AUC": 0.926423},
    {"Model": "Random Forest (CV)",  "Accuracy": 0.999687, "Precision": 0.989848, "Recall": 0.727612, "F1": 0.838710, "ROC_AUC": 0.984813, "PR_AUC": 0.916957},
    {"Model": "MLP (Keras)",         "Accuracy": 0.999487, "Precision": 0.993197, "Recall": 0.544776, "F1": 0.703614, "ROC_AUC": 0.979838, "PR_AUC": 0.741544},
    {"Model": "Decision Tree (CV)",  "Accuracy": 0.991592, "Precision": 0.110419, "Recall": 0.925373, "F1": 0.197295, "ROC_AUC": 0.961404, "PR_AUC": 0.622175},
    {"Model": "Logistic Regression", "Accuracy": 0.940196, "Precision": 0.017868, "Recall": 0.973881, "F1": 0.035092, "ROC_AUC": 0.984469, "PR_AUC": 0.584643},
]
df_metrics = pd.DataFrame(metrics)
best_row = df_metrics.sort_values(["F1", "ROC_AUC"], ascending=False).iloc[0]


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("🛡️ Fraud Risk Console")
st.sidebar.caption("Interactive PaySim fraud scoring")

threshold = st.sidebar.slider("Fraud decision threshold", 0.05, 0.95, 0.50, 0.01)

model_options = [
    "XGBoost (CV)",
    "Random Forest (CV)",
    "Decision Tree (CV)",
    "Logistic Regression",
]
if TF_AVAILABLE and assets.get("mlp") is not None:
    model_options.append("MLP (Keras)")

selected_model = st.sidebar.selectbox("Scoring model", model_options, index=0)

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
    default_step = 28
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
**Scenario:** {scenario}
"""
)

if not TF_AVAILABLE:
    st.sidebar.info("TensorFlow may be unavailable here, so MLP could be disabled.")


# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">PaySim Fraud Risk Console</div>
        <div class="hero-subtitle">
            A professional fraud analytics dashboard combining exploratory diagnostics, model benchmarking,
            explainability, and live transaction scoring for PaySim mobile money data.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Best Model</div><div class="metric-value">{best_row["Model"]}</div></div>',
        unsafe_allow_html=True
    )
with k2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Best F1</div><div class="metric-value">{best_row["F1"]:.3f}</div></div>',
        unsafe_allow_html=True
    )
with k3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">ROC-AUC</div><div class="metric-value">{best_row["ROC_AUC"]:.3f}</div></div>',
        unsafe_allow_html=True
    )
with k4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">PR-AUC</div><div class="metric-value">{best_row["PR_AUC"]:.3f}</div></div>',
        unsafe_allow_html=True
    )

st.write("")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Summary", "Descriptive Analytics", "Model Performance", "Prediction & Explainability"]
)


# ---------------------------------------------------
# Tab 1: Executive Summary
# ---------------------------------------------------
with tab1:
    left, right = st.columns([1.12, 1])

    with left:
        st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Business context, modeling approach, and key findings.</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="insight-box"><b>Problem:</b> Detect fraudulent mobile money transactions in a highly imbalanced classification setting where missing fraud is costly.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="insight-box"><b>Dataset:</b> PaySim synthetic financial transaction data with transaction type, balances, and engineered balance-delta features.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="insight-box"><b>Approach:</b> Evaluate multiple models, compare performance across imbalance-aware metrics, and support decisions with SHAP explainability.</div>',
            unsafe_allow_html=True
        )

    with right:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("#### Final Outcome")
        st.write(
            f"""
            **{best_row["Model"]}** delivered the best overall tradeoff across **F1**, **ROC-AUC**, and **PR-AUC**,
            making it the strongest candidate for deployment in this workflow.
            """
        )

        st.markdown("#### Why these metrics matter")
        st.write(
            """
            Fraud detection is dominated by class imbalance, so **accuracy alone is misleading**.
            This dashboard therefore emphasizes **Recall**, **F1**, and **PR-AUC**, which better reflect how well
            the model identifies rare fraud events without ignoring false positives.
            """
        )

        st.markdown("#### Dashboard Scope")
        st.write(
            """
            This application presents the full workflow required in the homework:
            descriptive analytics, model performance evaluation, explainability artifacts,
            and interactive transaction scoring.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------
# Tab 2: Descriptive Analytics
# ---------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">Descriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Exploratory findings from the transaction data before model training.</div>',
        unsafe_allow_html=True
    )

    eda1 = safe_path(MODELS_DIR, "eda_class_distribution.png")
    eda2 = safe_path(MODELS_DIR, "eda_transaction_type.png")
    eda3 = safe_path(MODELS_DIR, "eda_amount_distribution.png")
    eda4 = safe_path(MODELS_DIR, "eda_correlation_heatmap.png")

    c1, c2 = st.columns(2)

    with c1:
        if file_exists(eda1):
            st.image(eda1, caption="Target distribution: fraud is extremely rare", use_container_width=True)
        else:
            st.warning("Missing models/eda_class_distribution.png")

        if file_exists(eda3):
            st.image(eda3, caption="Log-scaled amount distribution by fraud label", use_container_width=True)
        else:
            st.warning("Missing models/eda_amount_distribution.png")

    with c2:
        if file_exists(eda2):
            st.image(eda2, caption="Fraud is concentrated in TRANSFER and CASH_OUT", use_container_width=True)
        else:
            st.warning("Missing models/eda_transaction_type.png")

        if file_exists(eda4):
            st.image(eda4, caption="Correlation heatmap across numeric features", use_container_width=True)
        else:
            st.warning("Missing models/eda_correlation_heatmap.png")

    st.markdown("#### Key Insights")
    st.markdown(
        """
- Fraud is a very small minority of transactions, confirming a severe class imbalance problem.
- Fraud is concentrated in **TRANSFER** and **CASH_OUT**, while other transaction types contribute minimally.
- Transaction amounts are highly skewed, making log-scale analysis more interpretable.
- Engineered balance delta features appear more informative than raw balances alone and support downstream classification.
        """
    )


# ---------------------------------------------------
# Tab 3: Model Performance
# ---------------------------------------------------
with tab3:
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Benchmark comparison across classifiers and evaluation diagnostics.</div>',
        unsafe_allow_html=True
    )

    if not (TF_AVAILABLE and assets.get("mlp") is not None):
        st.info("MLP metrics are included from training results, though live MLP scoring may be disabled in this deployment.")

    st.dataframe(
        df_metrics.style.highlight_max(subset=["F1", "ROC_AUC", "PR_AUC"], color="#dbeafe"),
        use_container_width=True,
    )

    st.markdown("#### Metric Comparison")
    chart_df = df_metrics.set_index("Model")[["F1", "ROC_AUC", "PR_AUC"]]
    st.bar_chart(chart_df)

    perf_left, perf_right = st.columns([1, 1])

    with perf_left:
        with st.expander("Best Hyperparameters", expanded=False):
            best_params_path = safe_path(MODELS_DIR, "best_params.json")
            if file_exists(best_params_path):
                try:
                    with open(best_params_path, "r", encoding="utf-8") as f:
                        st.code(f.read(), language="json")
                except Exception as e:
                    st.warning(f"Could not load best_params.json: {e}")
            else:
                st.info("Missing models/best_params.json")

    with perf_right:
        with st.expander("Performance Interpretation", expanded=False):
            st.markdown(
                """
- **XGBoost** leads overall due to its best combination of F1, ROC-AUC, and PR-AUC.
- **Random Forest** performs strongly and serves as a robust ensemble baseline.
- **Logistic Regression** provides a useful interpretable baseline but suffers from low precision.
- **Decision Tree** captures fraud aggressively but at the cost of weak precision and lower overall balance.
                """
            )

    with st.expander("ROC / Precision-Recall Curve Diagnostics", expanded=False):
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


# ---------------------------------------------------
# Tab 4: Prediction & Explainability
# ---------------------------------------------------
with tab4:
    left, right = st.columns([1.0, 1.02])

    with left:
        st.markdown('<div class="section-title">Interactive Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Run a live risk check under the selected model and threshold.</div>',
            unsafe_allow_html=True
        )

        with st.form("tx_form", clear_on_submit=False):
            step = st.number_input("Step", min_value=1, value=int(default_step), step=1)

            tx_type_options = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
            tx_type = st.selectbox(
                "Transaction Type",
                tx_type_options,
                index=tx_type_options.index(default_type)
            )

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

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Fraud Probability", format_pct(prob))
            with r2:
                st.metric("Threshold", f"{threshold:.2f}")
            with r3:
                st.metric("Model", selected_model)

            st.progress(min(max(prob, 0.0), 1.0))
            st.markdown(risk_badge(prob, threshold), unsafe_allow_html=True)

            action = recommended_action(prob, threshold)
            st.markdown(
                f'<div class="recommend-box"><b>Recommended action:</b> {action}</div>',
                unsafe_allow_html=True
            )

            with st.expander("Show model input row", expanded=False):
                st.dataframe(X_df, use_container_width=True)

            with st.expander("Scoring Interpretation", expanded=False):
                st.markdown(
                    """
- A score above the threshold is classified as **FRAUD**.
- Scores close to the threshold can be treated as manual-review candidates.
- This interface is intended as a decision-support prototype rather than a production control system.
                    """
                )

    with right:
        st.markdown('<div class="section-title">Explainability</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Global SHAP artifacts showing which features drive model predictions overall.</div>',
            unsafe_allow_html=True
        )

        i1, i2 = st.columns(2)
        with i1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Selected Model</div><div class="metric-value">{selected_model}</div></div>',
                unsafe_allow_html=True
            )
        with i2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Decision Threshold</div><div class="metric-value">{threshold:.2f}</div></div>',
                unsafe_allow_html=True
            )

        shap_summary = safe_path(MODELS_DIR, "shap_summary.png")
        shap_bar = safe_path(MODELS_DIR, "shap_bar.png")
        shap_waterfall = safe_path(MODELS_DIR, "shap_waterfall.png")

        if file_exists(shap_summary):
            st.image(shap_summary, caption="SHAP Summary Plot", use_container_width=True)
        else:
            st.warning("Missing models/shap_summary.png")

        if file_exists(shap_bar):
            st.image(shap_bar, caption="SHAP Global Feature Importance", use_container_width=True)
        else:
            st.warning("Missing models/shap_bar.png")

        with st.expander("Single-Prediction Waterfall Example", expanded=False):
            if file_exists(shap_waterfall):
                st.image(shap_waterfall, caption="SHAP Waterfall Plot", use_container_width=True)
            else:
                st.info("Missing models/shap_waterfall.png")
