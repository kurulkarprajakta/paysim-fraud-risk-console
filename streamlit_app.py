"""
Visa-Grade Fraud Risk Intelligence Platform
Built on PaySim fraud detection models by Prajakta Kurulkar
"""

import os
import json
import time
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    keras = None
    TF_AVAILABLE = False

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Risk Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — clean dark-navy enterprise look
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Top header bar */
.platform-header {
    background: linear-gradient(135deg, #0a1628 0%, #0f2447 100%);
    color: white;
    padding: 18px 28px;
    border-radius: 10px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid #1e3a5f;
}
.platform-title { font-size: 20px; font-weight: 600; letter-spacing: -0.3px; }
.platform-sub { font-size: 12px; color: #6b9bd2; margin-top: 2px; font-family: 'IBM Plex Mono', monospace; }
.platform-badge {
    background: #1a3a5c;
    border: 1px solid #2d5a8e;
    color: #5ba3d9;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
}

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 24px; }
.kpi-card {
    background: #0f2447;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px 20px;
    color: white;
}
.kpi-label { font-size: 11px; color: #6b9bd2; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 600; line-height: 1; }
.kpi-delta { font-size: 11px; margin-top: 4px; }
.kpi-up { color: #4ade80; }
.kpi-down { color: #f87171; }
.kpi-neutral { color: #94a3b8; }

/* Decision cards */
.decision-fraud {
    background: #2d0a0a;
    border: 2px solid #ef4444;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    color: #fca5a5;
}
.decision-legit {
    background: #0a1f0f;
    border: 2px solid #22c55e;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    color: #86efac;
}
.decision-label { font-size: 28px; font-weight: 700; letter-spacing: 2px; margin-bottom: 4px; }
.decision-prob { font-size: 36px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }

/* Rule cards */
.rule-card {
    background: #0f1e35;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #94c4f5;
}
.rule-card.triggered {
    border-left-color: #ef4444;
    background: #1a0f0f;
    color: #fca5a5;
}
.rule-card.warning {
    border-left-color: #f59e0b;
    background: #1a1400;
    color: #fcd34d;
}

/* Live feed rows */
.feed-row-fraud { background: rgba(239,68,68,0.08); border-left: 3px solid #ef4444; padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
.feed-row-legit { background: rgba(34,197,94,0.06); border-left: 3px solid #22c55e; padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }

/* Section headers */
.section-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

MODELS_DIR = "models"

# ─────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    preprocess = joblib.load(os.path.join(MODELS_DIR, "preprocess.pkl"))
    assets = {"preprocess": preprocess, "lr": None, "tree": None, "rf": None, "xgb": None, "mlp": None}
    for key, fname in [("lr","lr.pkl"),("tree","tree.pkl"),("rf","rf.pkl"),("xgb","xgb.pkl")]:
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            assets[key] = joblib.load(p)
    mlp_path = os.path.join(MODELS_DIR, "mlp.keras")
    if TF_AVAILABLE and os.path.exists(mlp_path):
        try:
            assets["mlp"] = keras.models.load_model(mlp_path)
        except Exception:
            pass
    return assets

try:
    assets = load_assets()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Core scoring helpers
# ─────────────────────────────────────────────
def make_input_df(tx_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, step=1):
    return pd.DataFrame([{
        "step": int(step),
        "type": tx_type,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest),
        "orig_balance_delta": float(oldbalanceOrg) - float(newbalanceOrig),
        "dest_balance_delta": float(newbalanceDest) - float(oldbalanceDest),
    }])

def score_transaction(model_key, X_df):
    preprocess = assets["preprocess"]
    Xp = preprocess.transform(X_df)
    model = assets[model_key]
    if model is None:
        raise RuntimeError(f"Model '{model_key}' not loaded.")
    if model_key == "mlp":
        X_dense = Xp.toarray() if hasattr(Xp, "toarray") else np.array(Xp)
        return float(model.predict(X_dense, verbose=0).ravel()[0])
    return float(model.predict_proba(Xp)[:, 1][0])

def score_all_models(X_df):
    results = {}
    model_map = {"lr": "Logistic Regression", "tree": "Decision Tree", "rf": "Random Forest", "xgb": "XGBoost"}
    if TF_AVAILABLE and assets.get("mlp"):
        model_map["mlp"] = "MLP (Keras)"
    for key, name in model_map.items():
        if assets.get(key):
            try:
                results[name] = score_transaction(key, X_df)
            except Exception:
                pass
    return results

# ─────────────────────────────────────────────
# Rule Engine
# ─────────────────────────────────────────────
RULES = [
    {
        "id": "R001",
        "name": "Full balance drain",
        "description": "Origin balance fully drained to zero",
        "severity": "HIGH",
        "check": lambda r, s: r["newbalanceOrig"] == 0 and r["oldbalanceOrg"] > 0,
        "action": "BLOCK",
    },
    {
        "id": "R002",
        "name": "High-value TRANSFER/CASH_OUT",
        "description": "Transaction amount exceeds $200,000",
        "severity": "HIGH",
        "check": lambda r, s: r["amount"] > 200_000 and r["type"] in ["TRANSFER", "CASH_OUT"],
        "action": "BLOCK",
    },
    {
        "id": "R003",
        "name": "Amount exceeds origin balance",
        "description": "Transaction amount larger than available balance",
        "severity": "HIGH",
        "check": lambda r, s: r["amount"] > r["oldbalanceOrg"] and r["type"] in ["TRANSFER", "CASH_OUT"],
        "action": "BLOCK",
    },
    {
        "id": "R004",
        "name": "ML ensemble score elevated",
        "description": "XGBoost fraud probability > 70%",
        "severity": "MEDIUM",
        "check": lambda r, s: s >= 0.70,
        "action": "STEP_UP_AUTH",
    },
    {
        "id": "R005",
        "name": "Zero destination balance",
        "description": "Destination had zero balance (mule account pattern)",
        "severity": "MEDIUM",
        "check": lambda r, s: r["oldbalanceDest"] == 0 and r["type"] in ["TRANSFER", "CASH_OUT"] and r["amount"] > 5000,
        "action": "REVIEW",
    },
    {
        "id": "R006",
        "name": "Off-hours large CASH_OUT",
        "description": "Large cash-out in early hours (step modulo 24 in 1-5)",
        "severity": "LOW",
        "check": lambda r, s: r["type"] == "CASH_OUT" and r["amount"] > 50_000 and (r["step"] % 24) in range(1, 6),
        "action": "ALERT",
    },
]

def evaluate_rules(row_dict, ml_score):
    triggered = []
    for rule in RULES:
        try:
            if rule["check"](row_dict, ml_score):
                triggered.append(rule)
        except Exception:
            pass
    return triggered

def final_decision(ml_score, triggered_rules, threshold):
    block_rules = [r for r in triggered_rules if r["action"] == "BLOCK"]
    if block_rules or ml_score >= threshold:
        return "FRAUD", "BLOCK"
    stepup = [r for r in triggered_rules if r["action"] == "STEP_UP_AUTH"]
    if stepup:
        return "SUSPICIOUS", "STEP_UP_AUTH"
    review = [r for r in triggered_rules if r["action"] == "REVIEW"]
    if review:
        return "REVIEW", "REVIEW"
    alert = [r for r in triggered_rules if r["action"] == "ALERT"]
    if alert:
        return "LOW RISK", "ALERT"
    return "LEGIT", "PASS"

# ─────────────────────────────────────────────
# Synthetic live transaction generator
# ─────────────────────────────────────────────
def generate_synthetic_transaction(force_fraud=False):
    tx_types = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
    fraud_types = ["TRANSFER", "CASH_OUT"]

    if force_fraud or random.random() < 0.15:
        tx_type = random.choice(fraud_types)
        oldbal = random.uniform(1000, 500000)
        amount = oldbal * random.uniform(0.85, 1.0)
        newbal_orig = max(0, oldbal - amount)
        oldbal_dest = 0.0
        newbal_dest = amount
        is_fraud = 1
    else:
        tx_type = random.choice(tx_types)
        oldbal = random.uniform(500, 100000)
        amount = random.uniform(10, min(oldbal * 0.5, 50000))
        newbal_orig = oldbal - amount if tx_type in ["TRANSFER","CASH_OUT","PAYMENT","DEBIT"] else oldbal + amount
        oldbal_dest = random.uniform(0, 50000)
        newbal_dest = oldbal_dest + amount
        is_fraud = 0

    step = random.randint(1, 743)
    return {
        "step": step,
        "type": tx_type,
        "amount": round(amount, 2),
        "oldbalanceOrg": round(oldbal, 2),
        "newbalanceOrig": round(newbal_orig, 2),
        "oldbalanceDest": round(oldbal_dest, 2),
        "newbalanceDest": round(newbal_dest, 2),
        "is_fraud_gt": is_fraud,
    }

# ─────────────────────────────────────────────
# Model performance data (from your HW run)
# ─────────────────────────────────────────────
MODEL_PERF = pd.DataFrame([
    {"Model": "XGBoost",          "Accuracy": 0.9998, "Precision": 0.9913, "Recall": 0.8507, "F1": 0.9157, "ROC_AUC": 0.9973, "PR_AUC": 0.9264},
    {"Model": "Random Forest",    "Accuracy": 0.9997, "Precision": 0.9898, "Recall": 0.7276, "F1": 0.8387, "ROC_AUC": 0.9848, "PR_AUC": 0.9170},
    {"Model": "MLP (Keras)",      "Accuracy": 0.9995, "Precision": 0.9932, "Recall": 0.5448, "F1": 0.7036, "ROC_AUC": 0.9798, "PR_AUC": 0.7415},
    {"Model": "Decision Tree",    "Accuracy": 0.9916, "Precision": 0.1104, "Recall": 0.9254, "F1": 0.1973, "ROC_AUC": 0.9614, "PR_AUC": 0.6222},
    {"Model": "Logistic Reg.",    "Accuracy": 0.9402, "Precision": 0.0179, "Recall": 0.9739, "F1": 0.0351, "ROC_AUC": 0.9845, "PR_AUC": 0.5846},
])

# ─────────────────────────────────────────────
# Session state for live feed
# ─────────────────────────────────────────────
if "feed" not in st.session_state:
    st.session_state.feed = []
if "feed_stats" not in st.session_state:
    st.session_state.feed_stats = {"total": 0, "fraud": 0, "blocked_value": 0.0}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="platform-header">
  <div>
    <div class="platform-title">🛡️ Fraud Risk Intelligence Platform</div>
    <div class="platform-sub">VADP-style · PaySim ML Engine · Rule + Model Decisioning</div>
  </div>
  <div class="platform-badge">SANDBOX · XGBoost v2.0 · 5 Models Loaded</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Strategy Controls")
    threshold = st.slider("ML fraud threshold", 0.05, 0.95, 0.50, 0.01,
                          help="Score above this → BLOCK decision")
    model_options = {"XGBoost (Best)": "xgb", "Random Forest": "rf",
                     "Logistic Regression": "lr", "Decision Tree": "tree"}
    if TF_AVAILABLE and assets.get("mlp"):
        model_options["MLP (Keras)"] = "mlp"
    selected_label = st.selectbox("Primary scoring model", list(model_options.keys()))
    selected_model = model_options[selected_label]

    st.divider()
    st.markdown("### 🔧 Rule Engine")
    rules_enabled = st.toggle("Enable rule layer", value=True)
    st.caption(f"{len(RULES)} rules active · R001–R006")

    st.divider()
    st.markdown("### 📊 Session Stats")
    stats = st.session_state.feed_stats
    st.metric("Transactions scored", stats["total"])
    st.metric("Fraud detected", stats["fraud"])
    st.metric("Value blocked", f"${stats['blocked_value']:,.0f}")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Score Transaction",
    "⚡ Live Feed",
    "📊 Model Performance",
    "🔍 Explainability (SHAP)",
    "📋 Rule Engine"
])

# ══════════════════════════════════════════════
# TAB 1 — Score a transaction
# ══════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown('<div class="section-title">Transaction Input</div>', unsafe_allow_html=True)

        # Quick-fill buttons
        qcol1, qcol2, qcol3 = st.columns(3)
        prefill = {}
        with qcol1:
            if st.button("🚨 Load Fraud Sample", use_container_width=True):
                prefill = {"type": "TRANSFER", "amount": 181480.25, "oldbalanceOrg": 181480.25,
                           "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 181480.25}
        with qcol2:
            if st.button("✅ Load Legit Sample", use_container_width=True):
                prefill = {"type": "PAYMENT", "amount": 1200.0, "oldbalanceOrg": 45000.0,
                           "newbalanceOrig": 43800.0, "oldbalanceDest": 5000.0, "newbalanceDest": 6200.0}
        with qcol3:
            if st.button("🎲 Random Transaction", use_container_width=True):
                tx = generate_synthetic_transaction()
                prefill = {k: tx[k] for k in ["type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]}
                prefill["step"] = tx["step"]

        with st.form("tx_form"):
            c1, c2 = st.columns(2)
            with c1:
                step = st.number_input("Step (time step)", min_value=1,
                                       value=int(prefill.get("step", 1)), step=1)
                tx_type = st.selectbox("Transaction type",
                                       ["TRANSFER","CASH_OUT","PAYMENT","CASH_IN","DEBIT"],
                                       index=["TRANSFER","CASH_OUT","PAYMENT","CASH_IN","DEBIT"].index(
                                           prefill.get("type","TRANSFER")))
                amount = st.number_input("Amount ($)", min_value=0.0,
                                         value=float(prefill.get("amount", 10000.0)), step=100.0,
                                         format="%.2f")
            with c2:
                oldbalanceOrg = st.number_input("Origin balance (before)",
                                                 min_value=0.0, value=float(prefill.get("oldbalanceOrg", 50000.0)),
                                                 step=100.0, format="%.2f")
                newbalanceOrig = st.number_input("Origin balance (after)",
                                                  min_value=0.0, value=float(prefill.get("newbalanceOrig", 40000.0)),
                                                  step=100.0, format="%.2f")
                oldbalanceDest = st.number_input("Destination balance (before)",
                                                  min_value=0.0, value=float(prefill.get("oldbalanceDest", 0.0)),
                                                  step=100.0, format="%.2f")
                newbalanceDest = st.number_input("Destination balance (after)",
                                                  min_value=0.0, value=float(prefill.get("newbalanceDest", 0.0)),
                                                  step=100.0, format="%.2f")

            submitted = st.form_submit_button("⚡ Score Transaction", use_container_width=True, type="primary")

    with col_right:
        st.markdown('<div class="section-title">Decision Output</div>', unsafe_allow_html=True)

        if submitted:
            X_df = make_input_df(tx_type, amount, oldbalanceOrg, newbalanceOrig,
                                  oldbalanceDest, newbalanceDest, step)
            row_dict = X_df.iloc[0].to_dict()

            with st.spinner("Scoring..."):
                try:
                    ml_score = score_transaction(selected_model, X_df)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

            triggered = evaluate_rules(row_dict, ml_score) if rules_enabled else []
            verdict, action = final_decision(ml_score, triggered, threshold)

            # Update session stats
            st.session_state.feed_stats["total"] += 1
            if verdict in ["FRAUD", "SUSPICIOUS"]:
                st.session_state.feed_stats["fraud"] += 1
                if action == "BLOCK":
                    st.session_state.feed_stats["blocked_value"] += amount

            # Decision card
            if verdict == "FRAUD":
                st.markdown(f"""
                <div class="decision-fraud">
                  <div class="decision-label">🚨 {verdict}</div>
                  <div class="decision-prob">{ml_score:.1%}</div>
                  <div style="font-size:13px;margin-top:8px;">Action: {action}</div>
                </div>""", unsafe_allow_html=True)
            elif verdict == "SUSPICIOUS":
                st.warning(f"⚠️ {verdict} · Score: {ml_score:.1%} · Action: {action}")
            elif verdict == "REVIEW":
                st.warning(f"🔍 {verdict} · Score: {ml_score:.1%} · Action: {action}")
            else:
                st.markdown(f"""
                <div class="decision-legit">
                  <div class="decision-label">✅ {verdict}</div>
                  <div class="decision-prob">{ml_score:.1%}</div>
                  <div style="font-size:13px;margin-top:8px;">Action: {action}</div>
                </div>""", unsafe_allow_html=True)

            st.divider()

            # All model scores
            st.markdown("**Ensemble scores (all models)**")
            all_scores = score_all_models(X_df)
            for mname, mscore in all_scores.items():
                color = "#ef4444" if mscore >= threshold else "#22c55e"
                bar_pct = int(mscore * 100)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                  <span style="font-size:12px;min-width:140px;color:#94a3b8;">{mname}</span>
                  <div style="flex:1;height:8px;background:#1e293b;border-radius:4px;">
                    <div style="width:{bar_pct}%;height:8px;background:{color};border-radius:4px;"></div>
                  </div>
                  <span style="font-size:12px;font-family:monospace;color:{color};min-width:40px;">{mscore:.1%}</span>
                </div>""", unsafe_allow_html=True)

            # Triggered rules
            if triggered:
                st.divider()
                st.markdown("**Rules triggered**")
                for rule in triggered:
                    sev_color = {"HIGH": "triggered", "MEDIUM": "warning", "LOW": ""}.get(rule["severity"], "")
                    st.markdown(f"""
                    <div class="rule-card {sev_color}">
                      [{rule['id']}] {rule['name']} · {rule['severity']} · → {rule['action']}<br>
                      <span style="opacity:0.7">{rule['description']}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.caption("No rules triggered.")

            # Derived features
            with st.expander("Derived features sent to model"):
                display_df = X_df.copy()
                display_df["orig_balance_delta"] = oldbalanceOrg - newbalanceOrig
                display_df["dest_balance_delta"] = newbalanceDest - oldbalanceDest
                st.dataframe(display_df, use_container_width=True)

        else:
            st.info("👈 Fill in transaction details and click **Score Transaction**, or use a Quick-Fill button above.")
            st.markdown("""
            **What happens when you score:**
            - Your trained XGBoost / RF / LR / DT / MLP models run inference
            - 6 fraud detection rules evaluate the transaction
            - A final decision is made: BLOCK / STEP_UP_AUTH / REVIEW / PASS
            - All model scores shown side by side
            """)

# ══════════════════════════════════════════════
# TAB 2 — Live Feed
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Simulated Transaction Feed</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        n_generate = st.number_input("Transactions to generate", 5, 100, 20, 5)
    with fc2:
        fraud_rate = st.slider("Injected fraud rate %", 5, 50, 15, 5)
    with fc3:
        st.write("")
        run_feed = st.button("▶ Run Feed", type="primary", use_container_width=True)

    if run_feed:
        progress = st.progress(0, text="Scoring transactions...")
        feed_rows = []

        for i in range(n_generate):
            force_fraud = random.random() < (fraud_rate / 100)
            tx = generate_synthetic_transaction(force_fraud=force_fraud)
            X_df = make_input_df(tx["type"], tx["amount"], tx["oldbalanceOrg"],
                                  tx["newbalanceOrig"], tx["oldbalanceDest"],
                                  tx["newbalanceDest"], tx["step"])
            try:
                score = score_transaction(selected_model, X_df)
            except Exception:
                score = 0.0

            triggered = evaluate_rules(tx, score) if rules_enabled else []
            verdict, action = final_decision(score, triggered, threshold)

            feed_rows.append({
                "txn_id": f"TXN-{random.randint(100000,999999)}",
                "type": tx["type"],
                "amount": tx["amount"],
                "score": score,
                "verdict": verdict,
                "action": action,
                "rules_hit": len(triggered),
                "actual_fraud": tx["is_fraud_gt"],
            })
            progress.progress((i + 1) / n_generate, text=f"Scored {i+1}/{n_generate}...")

        progress.empty()
        st.session_state.feed = feed_rows

        # Update stats
        fraud_count = sum(1 for r in feed_rows if r["verdict"] in ["FRAUD","SUSPICIOUS"])
        blocked_val = sum(r["amount"] for r in feed_rows if r["action"] == "BLOCK")
        st.session_state.feed_stats["total"] += n_generate
        st.session_state.feed_stats["fraud"] += fraud_count
        st.session_state.feed_stats["blocked_value"] += blocked_val

    if st.session_state.feed:
        feed_df = pd.DataFrame(st.session_state.feed)

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", len(feed_df))
        m2.metric("Flagged", int((feed_df["verdict"].isin(["FRAUD","SUSPICIOUS"])).sum()))
        m3.metric("Blocked value", f"${feed_df[feed_df['action']=='BLOCK']['amount'].sum():,.0f}")
        correct = ((feed_df["verdict"].isin(["FRAUD","SUSPICIOUS"])) == feed_df["actual_fraud"].astype(bool)).sum()
        m4.metric("Correct decisions", f"{correct}/{len(feed_df)}")

        st.divider()

        # Feed rows
        for _, row in feed_df.iterrows():
            is_fraud = row["verdict"] in ["FRAUD", "SUSPICIOUS"]
            css_class = "feed-row-fraud" if is_fraud else "feed-row-legit"
            icon = "🚨" if is_fraud else "✅"
            rules_txt = f"· {row['rules_hit']} rules" if row["rules_hit"] > 0 else ""
            st.markdown(f"""
            <div class="{css_class}">
              {icon} <b>{row['txn_id']}</b> &nbsp;|&nbsp;
              {row['type']} &nbsp;|&nbsp; ${row['amount']:,.0f} &nbsp;|&nbsp;
              Score: <b>{row['score']:.1%}</b> &nbsp;|&nbsp;
              <b>{row['verdict']}</b> → {row['action']} {rules_txt}
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("**Full results table**")

        display_df = feed_df.copy()
        display_df["score"] = display_df["score"].map(lambda x: f"{x:.1%}")
        display_df["amount"] = display_df["amount"].map(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)

    # Highlight best row
    def highlight_best(row):
        if row["Model"] == "XGBoost":
            return ["background-color: #0f2447; color: #5ba3d9; font-weight: bold"] * len(row)
        return [""] * len(row)

    perf_display = MODEL_PERF.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:
        perf_display[col] = perf_display[col].map(lambda x: f"{x:.4f}")

    st.dataframe(
        MODEL_PERF.style.apply(highlight_best, axis=1).format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1": "{:.4f}", "ROC_AUC": "{:.4f}", "PR_AUC": "{:.4f}",
        }),
        use_container_width=True, hide_index=True
    )

    st.divider()

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#0a0f1e")
    colors = ["#3b82f6", "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"]

    for ax, metric in zip(axes, ["F1", "ROC_AUC", "PR_AUC"]):
        ax.set_facecolor("#0f1629")
        bars = ax.barh(MODEL_PERF["Model"], MODEL_PERF[metric], color=colors, height=0.6)
        ax.set_xlim(0, 1.05)
        ax.set_title(metric.replace("_", " "), color="white", fontsize=12, pad=10)
        ax.tick_params(colors="white", labelsize=9)
        ax.spines[:].set_visible(False)
        for bar, val in zip(bars, MODEL_PERF[metric]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", color="white", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ROC/PR curve images from models folder
    st.markdown("**ROC & PR Curves (from training)**")
    curve_models = ["xgboost", "random_forest", "logistic_regression", "decision_tree"]
    roc_cols = st.columns(4)
    for col, m in zip(roc_cols, curve_models):
        roc_path = os.path.join(MODELS_DIR, f"{m}_roc_curve.png")
        if os.path.exists(roc_path):
            col.image(roc_path, caption=m.replace("_", " ").title(), use_container_width=True)

    pr_cols = st.columns(4)
    for col, m in zip(pr_cols, curve_models):
        pr_path = os.path.join(MODELS_DIR, f"{m}_pr_curve.png")
        if os.path.exists(pr_path):
            col.image(pr_path, caption=f"{m.replace('_',' ').title()} PR", use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — SHAP Explainability
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">SHAP Explainability — Why did the model flag this?</div>',
                unsafe_allow_html=True)

    sh1, sh2 = st.columns(2)
    shap_summary = os.path.join(MODELS_DIR, "shap_summary.png")
    shap_bar = os.path.join(MODELS_DIR, "shap_bar.png")
    shap_waterfall = os.path.join(MODELS_DIR, "shap_waterfall.png")

    with sh1:
        if os.path.exists(shap_summary):
            st.image(shap_summary, caption="SHAP Summary — feature impact distribution", use_container_width=True)
        else:
            st.warning("shap_summary.png not found in /models")

    with sh2:
        if os.path.exists(shap_bar):
            st.image(shap_bar, caption="SHAP Feature Importance (mean |SHAP|)", use_container_width=True)
        else:
            st.warning("shap_bar.png not found in /models")

    if os.path.exists(shap_waterfall):
        st.image(shap_waterfall, caption="SHAP Waterfall — single prediction explanation",
                 use_container_width=True)

    st.divider()
    st.markdown("**What SHAP tells us about fraud patterns**")
    insight_cols = st.columns(3)
    with insight_cols[0]:
        st.info("🔑 **orig_balance_delta** is the #1 driver — when a sender's balance is fully drained, the model weights this heavily toward fraud.")
    with insight_cols[1]:
        st.info("💡 **Transaction type** matters: TRANSFER and CASH_OUT dominate fraud cases. PAYMENT and DEBIT are near-zero fraud risk.")
    with insight_cols[2]:
        st.info("⚡ **dest_balance_delta** reveals mule accounts — destination going from zero to large values is a strong fraud signal.")

    # Live SHAP on a scored transaction
    if SHAP_AVAILABLE and assets.get("xgb"):
        st.divider()
        st.markdown("**Live SHAP waterfall — score a transaction to explain**")
        if st.button("🎲 Explain a random fraud transaction with SHAP"):
            tx = generate_synthetic_transaction(force_fraud=True)
            X_df = make_input_df(tx["type"], tx["amount"], tx["oldbalanceOrg"],
                                  tx["newbalanceOrig"], tx["oldbalanceDest"],
                                  tx["newbalanceDest"], tx["step"])
            try:
                preprocess = assets["preprocess"]
                Xp = preprocess.transform(X_df)
                if hasattr(Xp, "toarray"):
                    Xp_dense = Xp.toarray()
                else:
                    Xp_dense = np.array(Xp)

                explainer = shap.TreeExplainer(assets["xgb"])
                shap_values = explainer.shap_values(Xp_dense)

                feature_names = preprocess.get_feature_names_out() if hasattr(preprocess, "get_feature_names_out") else [f"f{i}" for i in range(Xp_dense.shape[1])]
                shap_df = pd.DataFrame({
                    "Feature": feature_names[:len(shap_values[0])],
                    "SHAP Value": shap_values[0]
                }).sort_values("SHAP Value", key=abs, ascending=False).head(10)

                fig2, ax2 = plt.subplots(figsize=(8, 4))
                fig2.patch.set_facecolor("#0a0f1e")
                ax2.set_facecolor("#0f1629")
                colors_shap = ["#ef4444" if v > 0 else "#22c55e" for v in shap_df["SHAP Value"]]
                ax2.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors_shap)
                ax2.set_title("Live SHAP values (XGBoost)", color="white", fontsize=12)
                ax2.tick_params(colors="white", labelsize=9)
                ax2.spines[:].set_visible(False)
                ax2.axvline(0, color="#475569", linewidth=0.8)
                red_patch = mpatches.Patch(color="#ef4444", label="→ Fraud")
                green_patch = mpatches.Patch(color="#22c55e", label="→ Legit")
                ax2.legend(handles=[red_patch, green_patch], labelcolor="white",
                           facecolor="#0f1629", edgecolor="#1e3a5f", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig2)
            except Exception as ex:
                st.warning(f"Live SHAP unavailable: {ex}. Pre-computed plots above are from your HW run.")

# ══════════════════════════════════════════════
# TAB 5 — Rule Engine
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Fraud Strategy Rule Engine</div>', unsafe_allow_html=True)

    st.markdown("""
    This rule layer sits **on top of** the ML model — mirroring how production fraud platforms like
    Visa Advanced Decisioning Platform (VADP) combine model scores with explicit business rules.
    Rules encode domain expertise that models may miss; models catch patterns rules can't hardcode.
    """)

    st.divider()

    # Rules table
    for rule in RULES:
        sev_color = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#3b82f6"}.get(rule["severity"], "#94a3b8")
        action_color = {"BLOCK": "#ef4444", "STEP_UP_AUTH": "#f59e0b",
                        "REVIEW": "#6366f1", "ALERT": "#3b82f6"}.get(rule["action"], "#94a3b8")
        st.markdown(f"""
        <div style="border:1px solid #1e3a5f;border-left:4px solid {sev_color};border-radius:8px;
                    padding:14px 18px;margin-bottom:10px;background:#0a0f1e;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="color:white;font-weight:600;font-size:14px;">[{rule['id']}] {rule['name']}</span>
            <span style="display:flex;gap:8px;">
              <span style="background:{sev_color}22;color:{sev_color};padding:2px 10px;border-radius:12px;font-size:11px;border:1px solid {sev_color}44;">{rule['severity']}</span>
              <span style="background:{action_color}22;color:{action_color};padding:2px 10px;border-radius:12px;font-size:11px;border:1px solid {action_color}44;">→ {rule['action']}</span>
            </span>
          </div>
          <div style="color:#64748b;font-size:13px;">{rule['description']}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**Decision hierarchy**")
    hierarchy_data = {
        "Priority": [1, 2, 3, 4, 5],
        "Condition": [
            "Any BLOCK rule triggered OR ML score ≥ threshold",
            "STEP_UP_AUTH rule triggered (e.g. ML score 70–threshold)",
            "REVIEW rule triggered",
            "ALERT rule triggered",
            "No rules triggered AND ML score < threshold",
        ],
        "Decision": ["FRAUD → BLOCK", "SUSPICIOUS → STEP_UP_AUTH",
                     "REVIEW → REVIEW QUEUE", "LOW RISK → ALERT", "LEGIT → PASS"],
    }
    st.dataframe(pd.DataFrame(hierarchy_data), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Test all rules with a fraud pattern**")
    if st.button("🧪 Run rule battery test"):
        test_cases = [
            {"label": "Full drain TRANSFER", "tx": {"type":"TRANSFER","amount":100000,"oldbalanceOrg":100000,"newbalanceOrig":0,"oldbalanceDest":0,"newbalanceDest":100000,"step":3}},
            {"label": "Normal PAYMENT", "tx": {"type":"PAYMENT","amount":500,"oldbalanceOrg":10000,"newbalanceOrig":9500,"oldbalanceDest":1000,"newbalanceDest":1500,"step":12}},
            {"label": "High-value CASH_OUT off-hours", "tx": {"type":"CASH_OUT","amount":300000,"oldbalanceOrg":400000,"newbalanceOrig":100000,"oldbalanceDest":5000,"newbalanceDest":305000,"step":2}},
            {"label": "Zero dest balance mule", "tx": {"type":"TRANSFER","amount":25000,"oldbalanceOrg":30000,"newbalanceOrig":5000,"oldbalanceDest":0,"newbalanceDest":25000,"step":15}},
        ]
        for tc in test_cases:
            tx = tc["tx"]
            X_df = make_input_df(tx["type"], tx["amount"], tx["oldbalanceOrg"],
                                  tx["newbalanceOrig"], tx["oldbalanceDest"],
                                  tx["newbalanceDest"], tx.get("step", 1))
            try:
                score = score_transaction(selected_model, X_df)
            except Exception:
                score = 0.0
            triggered = evaluate_rules(tx, score)
            verdict, action = final_decision(score, triggered, threshold)
            icon = "🚨" if verdict in ["FRAUD","SUSPICIOUS"] else "✅"
            rules_txt = ", ".join([r["id"] for r in triggered]) if triggered else "none"
            st.markdown(f"""
            <div class="feed-row-{'fraud' if verdict in ['FRAUD','SUSPICIOUS'] else 'legit'}">
              {icon} <b>{tc['label']}</b> · Score: {score:.1%} · Rules: {rules_txt} · <b>{verdict} → {action}</b>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#334155;font-size:12px;padding:8px 0;">
  Fraud Risk Intelligence Platform · Built on PaySim · Models: XGBoost, Random Forest, MLP, Decision Tree, Logistic Regression
  · SHAP Explainability · 6-Rule Strategy Engine · Prajakta Kurulkar · UW MSIS 2026
</div>
""", unsafe_allow_html=True)
