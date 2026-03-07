import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Optional TensorFlow / Keras
try:
    from tensorflow import keras  # type: ignore
    TF_AVAILABLE = True
except Exception:
    keras = None
    TF_AVAILABLE = False


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="PaySim Fraud Detection Workflow",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODELS_DIR = "models"


# =========================================================
# Styling
# =========================================================
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1240px;
            padding-top: 4.2rem;
            padding-bottom: 2rem;
        }

        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #172554 45%, #2563eb 100%);
            color: white;
            border-radius: 24px;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.16);
        }

        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: rgba(255,255,255,0.90);
            max-width: 980px;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 800;
            color: #0f172a;
            margin-top: 0.25rem;
            margin-bottom: 0.2rem;
        }

        .section-subtitle {
            color: #64748b;
            margin-bottom: 1rem;
        }

        .summary-box {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dbeafe;
            border-radius: 18px;
            padding: 1rem 1rem;
            margin-bottom: 0.9rem;
            color: #1e293b;
            box-shadow: 0 2px 10px rgba(15,23,42,0.04);
        }

        .takeaway-box {
            background: linear-gradient(180deg, #eff6ff 0%, #f8fbff 100%);
            border: 1px solid #93c5fd;
            border-radius: 18px;
            padding: 1rem 1rem;
            color: #1d4ed8;
            margin-top: 0.9rem;
            box-shadow: 0 2px 10px rgba(37,99,235,0.08);
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

        div[data-testid="stTabs"] button {
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 0.55rem 0.7rem;
            box-shadow: 0 2px 10px rgba(15,23,42,0.04);
        }

        div[data-testid="stMetric"] label {
            font-size: 0.78rem !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 0.95rem !important;
            line-height: 1.1 !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            font-size: 0.75rem !important;
        }

        .tiny-note {
            color: #64748b;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Helpers
# =========================================================
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


@st.cache_data
def load_metrics():
    csv_path = safe_path(MODELS_DIR, "model_comparison.csv")
    if file_exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
        rename_map = {
            "ROC AUC": "ROC_AUC",
            "PR AUC": "PR_AUC",
            "F1 Score": "F1",
        }
        df = df.rename(columns=rename_map)
        return df

    metrics = [
        {"Model": "XGBoost", "Accuracy": 0.999825, "Precision": 0.991304, "Recall": 0.850746, "F1": 0.915663, "ROC_AUC": 0.997263, "PR_AUC": 0.926423},
        {"Model": "Random Forest", "Accuracy": 0.999687, "Precision": 0.989848, "Recall": 0.727612, "F1": 0.838710, "ROC_AUC": 0.984813, "PR_AUC": 0.916957},
        {"Model": "MLP", "Accuracy": 0.999487, "Precision": 0.993197, "Recall": 0.544776, "F1": 0.703614, "ROC_AUC": 0.979838, "PR_AUC": 0.741544},
        {"Model": "Decision Tree", "Accuracy": 0.991592, "Precision": 0.110419, "Recall": 0.925373, "F1": 0.197295, "ROC_AUC": 0.961404, "PR_AUC": 0.622175},
        {"Model": "Logistic Regression", "Accuracy": 0.940196, "Precision": 0.017868, "Recall": 0.973881, "F1": 0.035092, "ROC_AUC": 0.984469, "PR_AUC": 0.584643},
    ]
    return pd.DataFrame(metrics)


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
    if model_name == "Logistic Regression":
        if assets["lr"] is None:
            raise FileNotFoundError("lr.pkl not found in /models.")
        return predict_proba(assets["lr"], X_df)

    if model_name == "Decision Tree":
        if assets["tree"] is None:
            raise FileNotFoundError("tree.pkl not found in /models.")
        return predict_proba(assets["tree"], X_df)

    if model_name == "Random Forest":
        if assets["rf"] is None:
            raise FileNotFoundError("rf.pkl not found in /models.")
        return predict_proba(assets["rf"], X_df)

    if model_name == "XGBoost":
        if assets["xgb"] is None:
            raise FileNotFoundError("xgb.pkl not found in /models.")
        return predict_proba(assets["xgb"], X_df)

    if model_name == "MLP":
        if (not TF_AVAILABLE) or (assets["mlp"] is None):
            raise RuntimeError("MLP is not available in this deployment.")
        preprocess = assets["preprocess"]
        Xp = preprocess_row(preprocess, X_df)
        X_dense = Xp.toarray() if hasattr(Xp, "toarray") else np.array(Xp)
        return float(assets["mlp"].predict(X_dense, verbose=0).ravel()[0])  # type: ignore

    raise ValueError(f"Unknown model: {model_name}")


def risk_label(prob: float, threshold: float = 0.5):
    return "FRAUD" if prob >= threshold else "LEGITIMATE"


def risk_badge(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return '<span class="risk-high">High Risk • Fraud</span>'
    if prob >= max(0.30, threshold - 0.15):
        return '<span class="risk-review">Moderate Risk • Review</span>'
    return '<span class="risk-low">Low Risk • Legitimate</span>'


def recommended_action(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "Escalate for fraud investigation."
    if prob >= max(0.30, threshold - 0.15):
        return "Send to manual review before approval."
    return "Approve transaction."


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# =========================================================
# Load assets and metrics
# =========================================================
try:
    assets = load_assets()
except Exception as e:
    st.error(f"Failed to load models or preprocessing assets: {e}")
    st.stop()

df_metrics = load_metrics().copy()

name_map = {
    "Decision Tree (CV)": "Decision Tree",
    "Random Forest (CV)": "Random Forest",
    "XGBoost (CV)": "XGBoost",
    "MLP (Keras)": "MLP",
}
if "Model" in df_metrics.columns:
    df_metrics["Model"] = df_metrics["Model"].replace(name_map)

metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"] if c in df_metrics.columns]
for col in metric_cols:
    df_metrics[col] = pd.to_numeric(df_metrics[col], errors="coerce")

best_row = df_metrics.sort_values(["F1", "ROC_AUC"], ascending=False).iloc[0]


# =========================================================
# Minimal sidebar
# =========================================================
st.sidebar.markdown("### PaySim Fraud Detection")
st.sidebar.caption("MSIS 522 • End-to-end data science workflow")
st.sidebar.markdown(
    """
This app presents:
- executive summary
- descriptive analytics
- model benchmarking
- explainability
- interactive prediction
"""
)
if not TF_AVAILABLE:
    st.sidebar.info("TensorFlow is unavailable here, so live MLP scoring may be disabled.")


# =========================================================
# Header
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">PaySim Fraud Detection Workflow</div>
        <div class="hero-subtitle">
            A clean end-to-end machine learning application for detecting fraudulent mobile money transactions using descriptive analytics,
            predictive modeling, explainability, and interactive scoring.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)


# =========================================================
# Tab 1 — Executive Summary
# =========================================================
with tab1:
    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A concise overview of the business problem, dataset, modeling approach, and final outcome.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="summary-box">
                <b>Problem</b><br>
                Fraudulent financial transactions are rare but costly. The goal of this project is to identify fraud accurately in a highly imbalanced transaction dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="summary-box">
                <b>Dataset</b><br>
                The analysis uses the PaySim synthetic mobile money transaction dataset, including transaction type, amount, account balances, and engineered balance-delta features.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="summary-box">
                <b>Prediction Task</b><br>
                This is a binary classification problem where the target variable indicates whether a transaction is fraudulent or legitimate.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="summary-box">
                <b>Approach</b><br>
                Multiple models were trained and compared, including Logistic Regression, Decision Tree, Random Forest, XGBoost, and a Multi-Layer Perceptron (MLP).
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="summary-box">
                <b>Key Result</b><br>
                <b>{best_row["Model"]}</b> achieved the strongest overall performance, with an F1 score of <b>{best_row["F1"]:.3f}</b>, ROC-AUC of <b>{best_row["ROC_AUC"]:.3f}</b>, and PR-AUC of <b>{best_row["PR_AUC"]:.3f}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="summary-box">
                <b>Why This Matters</b><br>
                In fraud detection, missing a fraudulent event can be expensive. That is why this project focuses on recall, F1 score, and PR-AUC rather than relying on accuracy alone.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="takeaway-box">
            <b>Final Takeaway:</b> Tree-based ensemble models performed best on this problem, and XGBoost was selected as the final model because it delivered the strongest balance between fraud detection capability and overall classification performance.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Tab 2 — Descriptive Analytics
# =========================================================
with tab2:
    st.markdown('<div class="section-title">Descriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Key exploratory findings used to understand the dataset before modeling.</div>',
        unsafe_allow_html=True
    )

    eda1 = safe_path(MODELS_DIR, "eda_class_distribution.png")
    eda2 = safe_path(MODELS_DIR, "eda_transaction_type.png")
    eda3 = safe_path(MODELS_DIR, "eda_amount_distribution.png")
    eda4 = safe_path(MODELS_DIR, "eda_correlation_heatmap.png")
    eda5 = safe_path(MODELS_DIR, "fraud_rate_over_time.png")

    # Row 1
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        if file_exists(eda1):
            st.image(eda1, caption="Target distribution", use_container_width=True)
            st.caption("Fraud cases represent a very small share of the dataset, confirming severe class imbalance.")
        else:
            st.warning("Missing models/eda_class_distribution.png")

    with r1c2:
        if file_exists(eda2):
            st.image(eda2, caption="Fraud rate by transaction type", use_container_width=True)
            st.caption("Fraud is concentrated mainly in TRANSFER and CASH_OUT transactions, making transaction type a strong signal.")
        else:
            st.warning("Missing models/eda_transaction_type.png")

    st.markdown("<div style='height: 0.7rem;'></div>", unsafe_allow_html=True)

    # Row 2
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        if file_exists(eda3):
            st.image(eda3, caption="Transaction amount distribution", use_container_width=True)
            st.caption("Transaction amounts are highly skewed, so a log-scale view makes fraud-related differences easier to interpret.")
        else:
            st.warning("Missing models/eda_amount_distribution.png")

    with r2c2:
        if file_exists(eda4):
            st.image(eda4, caption="Correlation heatmap", use_container_width=True)
            st.caption("Balance-related variables and engineered deltas show meaningful relationships that support predictive modeling.")
        else:
            st.warning("Missing models/eda_correlation_heatmap.png")

    st.markdown("<div style='height: 0.7rem;'></div>", unsafe_allow_html=True)

    # Row 3 - centered single chart
    st.markdown("#### Additional Descriptive View")
    center_left, center_col, center_right = st.columns([0.15, 0.70, 0.15])

    with center_col:
        if file_exists(eda5):
            st.image(eda5, caption="Fraud rate over time", use_container_width=True)
            st.caption("Fraud behavior varies across time steps, though these peaks should be interpreted cautiously because fraud is rare.")
        else:
            st.info("Optional plot missing: models/fraud_rate_over_time.png")

    st.markdown("#### Key Descriptive Findings")
    st.write(
        """
        The descriptive analysis shows that fraud detection in PaySim is an extremely imbalanced classification problem.
        Fraud is concentrated in a narrow subset of transaction types, and engineered features such as origin and destination
        balance deltas appear more informative than raw balances alone. These patterns justify the use of tree-based models
        and imbalance-aware evaluation metrics in later stages of the workflow.
        """
    )

# =========================================================
# Tab 3 — Model Performance
# =========================================================
with tab3:
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Comparison of all trained models using imbalance-aware classification metrics.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="summary-box">
            <b>How to read this section</b><br>
            The table below reports all evaluation metrics across the five models trained in this project.
            Because fraud detection is a highly imbalanced classification problem, the most informative metrics are
            <b>Recall</b>, <b>F1</b>, <b>ROC-AUC</b>, and <b>PR-AUC</b>. The comparison chart therefore emphasizes the
            key metrics that are most relevant for rare-event detection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clean table for display
    display_df = df_metrics.copy()

    if "Model" in display_df.columns:
        display_df = display_df[display_df["Model"].notna()]
        display_df = display_df[display_df["Model"].astype(str).str.strip() != ""]

    display_df = display_df.dropna(how="all")
    display_df = display_df.loc[:, ~display_df.columns.astype(str).str.contains(r"^Unnamed")]

    wanted_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    display_df = display_df[[c for c in wanted_cols if c in display_df.columns]]

    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(3)

    st.markdown("#### Model Comparison Table")
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

    st.markdown("<div style='height: 0.55rem;'></div>", unsafe_allow_html=True)

    st.markdown("#### Key Metric Comparison")
    st.caption(
        "This chart focuses on F1, ROC-AUC, and PR-AUC because these are the most informative metrics for highly imbalanced fraud detection problems."
    )

    chart_cols = [c for c in ["F1", "ROC_AUC", "PR_AUC"] if c in df_metrics.columns]
    chart_df = df_metrics.copy()
    chart_df = chart_df.loc[:, ~chart_df.columns.astype(str).str.contains(r"^Unnamed")]
    chart_df = chart_df.dropna(how="all")
    chart_df = chart_df.set_index("Model")[chart_cols]
    st.bar_chart(chart_df)

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    top_left, top_right = st.columns(2)

    with top_left:
        with st.expander("Best Hyperparameters", expanded=False):
            best_params_path = safe_path(MODELS_DIR, "best_params.json")
            if file_exists(best_params_path):
                try:
                    with open(best_params_path, "r", encoding="utf-8") as f:
                        best_params = json.load(f)
                    st.json(best_params)
                except Exception as e:
                    st.warning(f"Could not read best_params.json: {e}")
            else:
                st.info("Missing models/best_params.json")

    with top_right:
        with st.expander("Performance Interpretation", expanded=False):
            st.write(
                """
                XGBoost delivered the strongest overall performance and was selected as the final model for deployment.
                Random Forest also performed strongly and served as a robust ensemble benchmark.
                Logistic Regression and Decision Tree were useful baseline models, but they showed weaker balance across the key fraud-detection metrics.
                The MLP achieved good precision, though its recall and F1 remained below the best-performing tree-based ensemble.
                """
            )

    with st.expander("ROC and Precision-Recall Diagnostics", expanded=False):
        roc_lr = safe_path(MODELS_DIR, "logistic_regression_roc_curve.png")
        pr_lr = safe_path(MODELS_DIR, "logistic_regression_pr_curve.png")
        roc_dt = safe_path(MODELS_DIR, "decision_tree_roc_curve.png")
        pr_dt = safe_path(MODELS_DIR, "decision_tree_pr_curve.png")
        roc_rf = safe_path(MODELS_DIR, "random_forest_roc_curve.png")
        pr_rf = safe_path(MODELS_DIR, "random_forest_pr_curve.png")
        roc_xgb = safe_path(MODELS_DIR, "xgboost_roc_curve.png")
        pr_xgb = safe_path(MODELS_DIR, "xgboost_pr_curve.png")

        st.markdown("##### ROC Curves")
        roc_c1, roc_c2 = st.columns(2)

        with roc_c1:
            if file_exists(roc_xgb):
                st.image(roc_xgb, caption="XGBoost ROC Curve", use_container_width=True)
            if file_exists(roc_lr):
                st.image(roc_lr, caption="Logistic Regression ROC Curve", use_container_width=True)

        with roc_c2:
            if file_exists(roc_rf):
                st.image(roc_rf, caption="Random Forest ROC Curve", use_container_width=True)
            if file_exists(roc_dt):
                st.image(roc_dt, caption="Decision Tree ROC Curve", use_container_width=True)

        st.markdown("<div style='height: 0.55rem;'></div>", unsafe_allow_html=True)

        st.markdown("##### Precision-Recall Curves")
        pr_c1, pr_c2 = st.columns(2)

        with pr_c1:
            if file_exists(pr_xgb):
                st.image(pr_xgb, caption="XGBoost Precision-Recall Curve", use_container_width=True)
            if file_exists(pr_lr):
                st.image(pr_lr, caption="Logistic Regression Precision-Recall Curve", use_container_width=True)

        with pr_c2:
            if file_exists(pr_rf):
                st.image(pr_rf, caption="Random Forest Precision-Recall Curve", use_container_width=True)
            if file_exists(pr_dt):
                st.image(pr_dt, caption="Decision Tree Precision-Recall Curve", use_container_width=True)

    with st.expander("MLP Training History", expanded=False):
        mlp_loss = safe_path(MODELS_DIR, "mlp_loss_curve.png")
        mlp_auc = safe_path(MODELS_DIR, "mlp_auc_curve.png")

        hist_c1, hist_c2 = st.columns(2)

        with hist_c1:
            if file_exists(mlp_loss):
                st.image(mlp_loss, caption="MLP Loss Curve", use_container_width=True)
            else:
                st.info("Missing models/mlp_loss_curve.png")

        with hist_c2:
            if file_exists(mlp_auc):
                st.image(mlp_auc, caption="MLP AUC Curve", use_container_width=True)
            else:
                st.info("Missing models/mlp_auc_curve.png")

# =========================================================
# Tab 4 — Explainability & Interactive Prediction
# =========================================================
with tab4:
    st.markdown('<div class="section-title">Explainability & Interactive Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Use the trained models to score a custom transaction and review SHAP-based explanation artifacts.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="summary-box">
            <b>How to use this section</b><br>
            Select a model, choose a fraud threshold, and either use a quick scenario or enter custom transaction values manually.
            Then click <b>Run Prediction</b> to see the fraud probability, model decision, and recommended action.
            The SHAP plots below explain which features matter most overall and how they influence an example prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_col, result_col = st.columns(2)

    with control_col:
        st.markdown("#### Interactive Prediction")

        model_options = ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"]
        if TF_AVAILABLE and assets.get("mlp") is not None:
            model_options.append("MLP")

        selected_model = st.selectbox("Model", model_options, index=0)
        threshold = st.slider("Fraud decision threshold", 0.05, 0.95, 0.50, 0.01)

        scenario = st.selectbox(
            "Quick scenario",
            ["Custom", "High-risk transfer", "Medium-risk cash-out", "Low-risk payment"],
            index=0,
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

        with st.form("prediction_form", clear_on_submit=False):
            step = st.number_input("Step", min_value=1, value=int(default_step), step=1)

            tx_type_options = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
            tx_type = st.selectbox(
                "Transaction Type",
                tx_type_options,
                index=tx_type_options.index(default_type),
            )

            amount = st.number_input("Amount", min_value=0.0, value=float(default_amount), step=100.0)
            oldbalanceOrg = st.number_input("Origin old balance", min_value=0.0, value=float(default_old_org), step=100.0)
            newbalanceOrig = st.number_input("Origin new balance", min_value=0.0, value=float(default_new_org), step=100.0)
            oldbalanceDest = st.number_input("Destination old balance", min_value=0.0, value=float(default_old_dest), step=100.0)
            newbalanceDest = st.number_input("Destination new balance", min_value=0.0, value=float(default_new_dest), step=100.0)

            submitted = st.form_submit_button("Run Prediction")

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
                st.session_state["last_model"] = selected_model
                st.session_state["last_threshold"] = threshold
            except Exception as e:
                st.error(f"Scoring failed: {e}")

    with result_col:
        st.markdown("#### Prediction Output")

        if "last_prob" in st.session_state:
            prob = st.session_state["last_prob"]
            decision = st.session_state["last_decision"]
            X_df = st.session_state["last_input"]
            last_model = st.session_state["last_model"]
            last_threshold = st.session_state["last_threshold"]

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Fraud Probability", format_pct(prob))
            with m2:
                st.metric("Decision", decision)
            with m3:
                st.metric("Model Used", last_model)

            st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
            st.progress(min(max(prob, 0.0), 1.0))
            st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
            st.markdown(risk_badge(prob, last_threshold), unsafe_allow_html=True)

            st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="takeaway-box">
                    <b>Recommended Action:</b> {recommended_action(prob, last_threshold)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height: 0.55rem;'></div>", unsafe_allow_html=True)

            with st.expander("Input row used for prediction", expanded=False):
                st.dataframe(X_df, use_container_width=True)

            st.caption(
                "A transaction is classified as fraud when its predicted probability exceeds the selected threshold."
            )
        else:
            st.info("Enter transaction values and run a prediction to view the model output.")

    st.markdown("---")
    st.markdown("#### Explainability")

    st.write(
        """
        The plots below help interpret the trained XGBoost model.
        The summary plot shows how features influence predictions across many transactions,
        the bar plot ranks the most important drivers globally, and the waterfall plot explains one example prediction step by step.
        """
    )

    shap_summary = safe_path(MODELS_DIR, "shap_summary.png")
    shap_bar = safe_path(MODELS_DIR, "shap_bar.png")
    shap_waterfall = safe_path(MODELS_DIR, "shap_waterfall.png")

    s1, s2 = st.columns(2)

    with s1:
        if file_exists(shap_summary):
            st.image(shap_summary, caption="SHAP Summary Plot", use_container_width=True)
        else:
            st.warning("Missing models/shap_summary.png")

    with s2:
        if file_exists(shap_bar):
            st.image(shap_bar, caption="SHAP Feature Importance Bar Plot", use_container_width=True)
        else:
            st.warning("Missing models/shap_bar.png")

    st.markdown("#### Waterfall Example")
    wf_left, wf_center, wf_right = st.columns([0.12, 0.76, 0.12])

    with wf_center:
        if file_exists(shap_waterfall):
            st.image(
                shap_waterfall,
                caption="SHAP Waterfall Plot for an Example Prediction",
                use_container_width=True
            )
            st.caption("This waterfall plot explains one representative prediction from the trained XGBoost model.")
        else:
            st.info("Missing models/shap_waterfall.png")




