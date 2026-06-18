"""Frontend Streamlit : detecteur de fraude Ethereum."""
from __future__ import annotations

import os
from datetime import datetime

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ethereum_fraud.config import (
    CATEGORICAL_FEATURES,
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    NUMERIC_FEATURES,
)
from ethereum_fraud.tracking import (
    get_all_runs,
    get_latest_confusion_matrix,
    get_latest_metrics,
    get_latest_shap,
)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
API_EXTERNAL_URL = os.environ.get("API_EXTERNAL_URL", API_URL)
MLFLOW_EXTERNAL_URL = os.environ.get("MLFLOW_EXTERNAL_URL", "http://localhost:5000")

st.set_page_config(
    page_title="Ethereum Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
header[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

/* ── Hero ─────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 45%, #2563EB 80%, #0EA5E9 100%);
    padding: 2.5rem 3rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(37,99,235,0.3);
}
.hero::before {
    content: "";
    position: absolute;
    top: -60%; right: -5%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: #BAE6FD;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
    margin-bottom: 1rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.hero h1 {
    color: white; margin: 0 0 0.75rem 0;
    font-size: 2.4rem; font-weight: 800;
}
.hero p {
    color: #BFDBFE; margin: 0 0 2rem 0;
    font-size: 1rem; line-height: 1.7; max-width: 580px;
}
.hero-stats { display: flex; gap: 3rem; flex-wrap: wrap; }
.hero-stat-value { font-size: 2rem; font-weight: 800; color: white; display: block; }
.hero-stat-label { font-size: 0.76rem; color: #93C5FD; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── KPI cards ────────────────────────────────────────── */
.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem 1.25rem;
    text-align: center;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
}
.kpi-icon  { font-size: 1.9rem; }
.kpi-value { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin: 0.3rem 0; line-height: 1; }
.kpi-label { font-size: 0.76rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Info cards ───────────────────────────────────────── */
.info-card {
    background: white;
    border-radius: 16px;
    padding: 1.75rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    height: 100%;
    margin-bottom: 1rem;
}

/* ── Section title ───────────────────────────────────── */
.section-title {
    font-size: 1.15rem; font-weight: 700; color: #1E3A8A;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #BFDBFE;
    margin-bottom: 1.25rem;
}

/* ── Tech badges ─────────────────────────────────────── */
.tech-badge {
    display: inline-block;
    background: #EFF6FF; border: 1px solid #BFDBFE;
    color: #1E40AF; padding: 0.4rem 0.9rem;
    border-radius: 8px; font-size: 0.82rem; font-weight: 600;
    margin: 0.2rem;
}

/* ── Metric containers ───────────────────────────────── */
[data-testid="metric-container"] {
    background: white; border: 1px solid #E2E8F0;
    border-radius: 14px; padding: 1.25rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E3A8A 65%, #1E40AF 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] [data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
[data-testid="stSidebar"] [data-testid="stLinkButton"] a {
    background: rgba(255,255,255,0.12) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 10px; font-weight: 600;
}
[data-testid="stSidebar"] [data-testid="stLinkButton"] a:hover {
    background: rgba(255,255,255,0.22) !important;
}

/* ── Tabs ────────────────────────────────────────────── */
[data-testid="stTabs"] button { font-weight: 700; font-size: 0.95rem; }

/* ── Result boxes ────────────────────────────────────── */
.result-fraud {
    background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
    border: 2px solid #FB7185; border-radius: 20px; padding: 2rem;
    text-align: center; box-shadow: 0 8px 32px rgba(251,113,133,0.2);
}
.result-legit {
    background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
    border: 2px solid #4ADE80; border-radius: 20px; padding: 2rem;
    text-align: center; box-shadow: 0 8px 32px rgba(74,222,128,0.15);
}
.result-emoji { font-size: 3.5rem; display: block; margin-bottom: 0.5rem; }
.result-title { font-size: 1.6rem; font-weight: 800; }

/* ── Best run ────────────────────────────────────────── */
.run-best {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border: 2px solid #3B82F6; border-radius: 14px;
    padding: 1.25rem 1.5rem; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_health() -> dict:
    try:
        return httpx.get(f"{API_URL}/health", timeout=2.0).json()
    except Exception:
        return {}


def fetch_model_info() -> dict:
    try:
        return httpx.get(f"{API_URL}/model-info", timeout=2.0).json()
    except Exception:
        return {}


@st.cache_data(ttl=10)
def fetch_predictions_log() -> list[dict]:
    try:
        return httpx.get(f"{API_URL}/predictions", timeout=5.0).json()
    except Exception:
        return []


@st.cache_data(ttl=60)
def cached_metrics() -> dict:
    try:
        return get_latest_metrics()
    except Exception:
        return {}


@st.cache_data(ttl=60)
def cached_runs() -> list[dict]:
    try:
        return get_all_runs()
    except Exception:
        return []


@st.cache_data(ttl=60)
def cached_confusion_matrix() -> bytes | None:
    try:
        return get_latest_confusion_matrix()
    except Exception:
        return None


@st.cache_data(ttl=60)
def cached_shap() -> bytes | None:
    try:
        return get_latest_shap()
    except Exception:
        return None


def make_gauge(prob: float) -> go.Figure:
    color = "#EF4444" if prob > 0.6 else "#F59E0B" if prob > 0.3 else "#10B981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        title={"text": "Score de risque", "font": {"size": 15, "color": "#475569"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1", "dtick": 25},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#F8FAFC",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "#DCFCE7"},
                {"range": [30, 60], "color": "#FEF9C3"},
                {"range": [60, 100],"color": "#FEE2E2"},
            ],
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_metrics_chart(runs: list[dict]) -> go.Figure:
    names = [r["name"] for r in reversed(runs)]
    f1s   = [r["metrics"].get("f1", 0) for r in reversed(runs)]
    rocs  = [r["metrics"].get("roc_auc", 0) for r in reversed(runs)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=names, y=f1s, name="F1",
        mode="lines+markers",
        line=dict(color="#3B82F6", width=2.5),
        marker=dict(size=8, color="#3B82F6", line=dict(color="white", width=1.5)),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=names, y=rocs, name="ROC-AUC",
        mode="lines+markers",
        line=dict(color="#10B981", width=2.5),
        marker=dict(size=8, color="#10B981", line=dict(color="white", width=1.5)),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.07)",
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=60),
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        yaxis=dict(range=[0, 1.05], gridcolor="#E2E8F0", tickformat=".2f"),
        xaxis=dict(gridcolor="#E2E8F0", tickangle=-30),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# BENHADDAD Lamia")
    st.caption("Master 2 ESGI · MLOps")
    st.divider()

    health = fetch_health()
    model_ready = health.get("model_ready", False)
    dot = "🟢" if model_ready else "🔴"
    label = "API opérationnelle" if model_ready else "API indisponible"
    st.markdown(f"{dot} **{label}**")

    st.markdown("")
    metrics = cached_metrics()
    if metrics:
        c1, c2 = st.columns(2)
        c1.metric("F1", f"{metrics.get('f1', 0):.3f}")
        c2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

    st.divider()
    st.link_button("🔬 Ouvrir MLflow", MLFLOW_EXTERNAL_URL, use_container_width=True)
    st.link_button("🌀 Ouvrir Airflow", "http://141.253.125.253:8080", use_container_width=True)
    st.link_button("📖 API Documentation", f"{API_EXTERNAL_URL}/docs", use_container_width=True)
    st.link_button("🐙 GitHub", "https://github.com/Lamia-26/Ethereum-fraud-detection", use_container_width=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_accueil, tab_predict, tab_experiments, tab_monitoring, tab_trace = st.tabs([
    "🏠  Accueil", "🔍  Prédiction", "📊  Expériences", "📈  Monitoring", "📋  Traçabilité",
])


# ═════════════════════════════════════════════════════════════════════════════
# ACCUEIL
# ═════════════════════════════════════════════════════════════════════════════
with tab_accueil:
    metrics = cached_metrics()
    f1_val  = metrics.get("f1", 0)
    roc_val = metrics.get("roc_auc", 0)

    st.markdown(f"""
    <div class="hero">
      <div class="hero-badge">🔗 Blockchain · Machine Learning · MLOps</div>
      <h1>Détection de fraude sur Ethereum</h1>
      <p>Pipeline MLOps complet pour détecter automatiquement les transactions frauduleuses
      sur la blockchain Ethereum à partir de caractéristiques comportementales des portefeuilles.</p>
      <div class="hero-stats">
        <div>
          <span class="hero-stat-value">47</span>
          <span class="hero-stat-label">Features</span>
        </div>
        <div>
          <span class="hero-stat-value">3</span>
          <span class="hero-stat-label">Modèles comparés</span>
        </div>
        <div>
          <span class="hero-stat-value">{f1_val:.3f}</span>
          <span class="hero-stat-label">F1 Score</span>
        </div>
        <div>
          <span class="hero-stat-value">{roc_val:.3f}</span>
          <span class="hero-stat-label">ROC-AUC</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="kpi-card">
      <div class="kpi-icon">🔢</div>
      <div class="kpi-value">{len(NUMERIC_FEATURES)}</div>
      <div class="kpi-label">Features numériques</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="kpi-card">
      <div class="kpi-icon">🏷️</div>
      <div class="kpi-value">{len(CATEGORICAL_FEATURES)}</div>
      <div class="kpi-label">Features catégorielles</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="kpi-card">
      <div class="kpi-icon">🤖</div>
      <div class="kpi-value">3</div>
      <div class="kpi-label">Algorithmes comparés</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="kpi-card">
      <div class="kpi-icon">🎯</div>
      <div class="kpi-value">FLAG</div>
      <div class="kpi-label">Variable cible (0 / 1)</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🧠 Modèle en production</p>', unsafe_allow_html=True)
        info = fetch_model_info()
        st.markdown(f"- **Algorithme** : LightGBM (meilleur run GridSearchCV)")
        st.markdown(f"- **Version registry** : {info.get('version', '—')}")
        if metrics:
            st.markdown(f"- **F1 Score** : `{metrics.get('f1', 0):.4f}`")
            st.markdown(f"- **ROC-AUC** : `{metrics.get('roc_auc', 0):.4f}`")
        st.markdown(f"- **Seuil F1 min** : `{EVAL_F1_MIN}`")
        st.markdown(f"- **Seuil ROC-AUC min** : `{EVAL_ROC_AUC_MIN}`")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">⚙️ Stack technique</p>', unsafe_allow_html=True)
        techs = [
            "🔬 MLflow", "⚡ FastAPI", "📊 Streamlit", "🐳 Docker Compose",
            "🔄 GitHub Actions", "☁️ Oracle Cloud", "🤖 XGBoost", "🌿 LightGBM",
            "🌲 Random Forest", "🔍 SHAP", "📦 scikit-learn", "🐍 Python 3.13",
        ]
        html = '<div style="display:flex;flex-wrap:wrap;gap:0.35rem;margin-top:0.5rem;">'
        for t in techs:
            html += f'<span class="tech-badge">{t}</span>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        with st.expander(f"📋 Features numériques ({len(NUMERIC_FEATURES)})"):
            cols = st.columns(2)
            for i, feat in enumerate(NUMERIC_FEATURES):
                cols[i % 2].markdown(f"- `{feat}`")
    with cb:
        with st.expander(f"🏷️ Features catégorielles ({len(CATEGORICAL_FEATURES)})"):
            for feat in CATEGORICAL_FEATURES:
                st.markdown(f"- `{feat}`")


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("## 🔍 Analyser une transaction Ethereum")
    st.markdown("Renseignez les caractéristiques du portefeuille pour obtenir un score de risque.")
    st.markdown("")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**💸 Transactions envoyées**")
            sent_tnx             = st.number_input("Sent tnx", min_value=0.0, value=0.0)
            avg_min_sent         = st.number_input("Avg min between sent tnx", min_value=0.0, value=0.0)
            total_sent           = st.number_input("total Ether sent", min_value=0.0, value=0.0)
            total_sent_contracts = st.number_input("total ether sent contracts", min_value=0.0, value=0.0)

        with col2:
            st.markdown("**📥 Transactions reçues**")
            received_tnx     = st.number_input("Received Tnx", min_value=0.0, value=0.0)
            avg_min_received = st.number_input("Avg min between received tnx", min_value=0.0, value=0.0)
            total_received   = st.number_input("total ether received", min_value=0.0, value=0.0)
            total_balance    = st.number_input("total ether balance", value=0.0)

        with col3:
            st.markdown("**⏱️ Activité & Contrats**")
            time_diff    = st.number_input("Time Diff between first and last (Mins)", min_value=0.0, value=0.0)
            nb_contracts = st.number_input("Number of Created Contracts", min_value=0.0, value=0.0)
            st.markdown("**🪙 ERC20**")
            erc20_most_sent = st.selectbox("Most sent token type", ["", "ERC20", "ERC721", "ERC1155", "other"])
            erc20_most_rec  = st.selectbox("Most rec token type",  ["", "ERC20", "ERC721", "ERC1155", "other"])

        submitted = st.form_submit_button("🔍 Analyser la transaction", use_container_width=True)

    if submitted:
        payload = {
            "Sent tnx": sent_tnx,
            "Received Tnx": received_tnx,
            "Avg min between sent tnx": avg_min_sent,
            "Avg min between received tnx": avg_min_received,
            "Time Diff between first and last (Mins)": time_diff,
            "Number of Created Contracts": nb_contracts,
            "total Ether sent": total_sent,
            "total ether received": total_received,
            "total ether balance": total_balance,
            "total ether sent contracts": total_sent_contracts,
            " ERC20 most sent token type": erc20_most_sent or None,
            " ERC20_most_rec_token_type": erc20_most_rec or None,
        }
        try:
            response = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
            st.session_state.last_result = {
                "prediction": result["prediction"],
                "probability": result["probability"],
                "feedback": None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "payload": payload,
            }
            st.session_state.history.insert(0, st.session_state.last_result)
        except httpx.HTTPError as exc:
            st.error(f"Erreur API : {exc}")
            st.session_state.last_result = None

    if st.session_state.last_result:
        res        = st.session_state.last_result
        prediction = res["prediction"]
        probability = res["probability"]

        st.divider()
        col_gauge, col_result = st.columns([1, 1], gap="large")

        with col_gauge:
            st.plotly_chart(make_gauge(probability), use_container_width=True)

        with col_result:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-fraud">
                  <span class="result-emoji">🚨</span>
                  <div class="result-title" style="color:#DC2626;">Transaction frauduleuse</div>
                  <p style="color:#991B1B;margin-top:0.5rem;">Ce portefeuille présente un risque élevé de fraude.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                  <span class="result-emoji">✅</span>
                  <div class="result-title" style="color:#16A34A;">Transaction légitime</div>
                  <p style="color:#15803D;margin-top:0.5rem;">Ce portefeuille semble normal.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Probabilité de fraude", f"{probability:.1%}")
            st.metric("Heure", res["timestamp"])

        st.divider()
        st.markdown("**Cette prédiction est-elle correcte ?**")
        cy, cn = st.columns(2)
        if cy.button("👍 Correcte", use_container_width=True):
            st.session_state.history[0]["feedback"] = "correct"
            st.success("Merci pour votre retour !")
        if cn.button("👎 Incorrecte", use_container_width=True):
            st.session_state.history[0]["feedback"] = "incorrect"
            st.warning("Merci, nous en tiendrons compte.")

        if st.session_state.history:
            st.divider()
            st.markdown("### 📋 Historique de session")
            rows = [
                {
                    "Heure": h["timestamp"],
                    "Résultat": "🚨 Fraude" if h["prediction"] == 1 else "✅ Légitime",
                    "Probabilité": f"{h['probability']:.1%}",
                    "Feedback": h.get("feedback") or "—",
                }
                for h in st.session_state.history
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if st.button("🗑️ Effacer l'historique"):
                st.session_state.history = []
                st.session_state.last_result = None
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIENCES
# ═════════════════════════════════════════════════════════════════════════════
with tab_experiments:
    st.markdown("## 📊 Historique des expériences MLflow")

    runs = cached_runs()
    if not runs:
        st.info("Aucune expérience trouvée. Lancez un entraînement d'abord.")
    else:
        best = max(runs, key=lambda r: r["metrics"].get("roc_auc", 0))
        c1, c2, c3 = st.columns(3)
        c1.metric("Total runs", len(runs))
        c2.metric("Meilleur ROC-AUC", f"{best['metrics'].get('roc_auc', 0):.4f}")
        c3.metric("Meilleur F1", f"{best['metrics'].get('f1', 0):.4f}")

        st.divider()
        st.markdown('<p class="section-title">📈 Évolution des métriques</p>', unsafe_allow_html=True)
        st.plotly_chart(make_metrics_chart(runs), use_container_width=True)

        st.divider()
        st.markdown('<p class="section-title">🏆 Meilleur run</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="run-best">
          🏆 <strong>{best['name']}</strong> &nbsp;·&nbsp;
          F1 = <strong>{best['metrics'].get('f1', 0):.4f}</strong> &nbsp;·&nbsp;
          ROC-AUC = <strong>{best['metrics'].get('roc_auc', 0):.4f}</strong> &nbsp;·&nbsp;
          {best['date']}
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown('<p class="section-title">📋 Tous les runs</p>', unsafe_allow_html=True)
        rows_exp = [
            {
                "Date": r["date"],
                "Run": r["name"],
                "Statut": r["status"],
                "F1": round(r["metrics"].get("f1", 0), 4),
                "ROC-AUC": round(r["metrics"].get("roc_auc", 0), 4),
            }
            for r in runs
        ]
        st.dataframe(pd.DataFrame(rows_exp), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown('<p class="section-title">🔲 Matrice de confusion (dernier run)</p>', unsafe_allow_html=True)
        cm_bytes = cached_confusion_matrix()
        if cm_bytes:
            col_img, _ = st.columns([1, 1])
            col_img.image(cm_bytes, width=420)
        else:
            st.info("Matrice de confusion non disponible.")


# ═════════════════════════════════════════════════════════════════════════════
# MONITORING
# ═════════════════════════════════════════════════════════════════════════════
with tab_monitoring:
    st.markdown("## 📈 Monitoring du modèle")

    health      = fetch_health()
    metrics     = cached_metrics()
    model_ready = health.get("model_ready", False)

    if model_ready and metrics:
        f1_ok  = metrics.get("f1", 0) >= EVAL_F1_MIN
        roc_ok = metrics.get("roc_auc", 0) >= EVAL_ROC_AUC_MIN
        if f1_ok and roc_ok:
            st.success("✅ Le modèle est en production et respecte tous les seuils de qualité.")
        else:
            st.warning("⚠️ Le modèle est en production mais certains seuils ne sont pas atteints.")
    else:
        st.error("❌ Le modèle n'est pas disponible.")

    st.divider()
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<p class="section-title">🎯 Performance du modèle</p>', unsafe_allow_html=True)
        if metrics:
            f1  = metrics.get("f1", 0)
            roc = metrics.get("roc_auc", 0)
            st.markdown(f"**F1 Score** — seuil minimum : `{EVAL_F1_MIN}`")
            st.progress(f1, text=f"{f1:.4f}")
            st.markdown("")
            st.markdown(f"**ROC-AUC** — seuil minimum : `{EVAL_ROC_AUC_MIN}`")
            st.progress(roc, text=f"{roc:.4f}")
        else:
            st.info("Métriques MLflow inaccessibles.")

    with col2:
        st.markdown('<p class="section-title">🖥️ Statut des services</p>', unsafe_allow_html=True)
        st.metric("Statut API", "✅ En ligne" if model_ready else "❌ Hors ligne")

        total   = len(st.session_state.history)
        fraudes = sum(1 for h in st.session_state.history if h["prediction"] == 1)
        st.metric("Prédictions (session)", total)
        if total > 0:
            st.metric("Fraudes détectées", f"{fraudes} / {total} ({fraudes / total:.0%})")

        st.markdown("<br>", unsafe_allow_html=True)
        st.link_button("📖 API Documentation", f"{API_EXTERNAL_URL}/docs", use_container_width=True)

    st.divider()
    st.markdown('<p class="section-title">🖼️ Artefacts du dernier run</p>', unsafe_allow_html=True)

    col_cm, col_shap = st.columns(2, gap="large")
    with col_cm:
        st.markdown("**🔲 Matrice de confusion**")
        cm_bytes = cached_confusion_matrix()
        if cm_bytes:
            st.image(cm_bytes, use_container_width=True)
        else:
            st.info("Non disponible.")
    with col_shap:
        st.markdown("**🔍 Importance des variables (SHAP)**")
        shap_bytes = cached_shap()
        if shap_bytes:
            st.image(shap_bytes, use_container_width=True)
        else:
            st.info("Non disponible — lancez `make train-models`.")

    st.divider()
    st.link_button("🔬 Ouvrir MLflow UI", MLFLOW_EXTERNAL_URL, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TRACABILITE
# ═════════════════════════════════════════════════════════════════════════════
with tab_trace:
    st.markdown("## 📋 Traçabilité des prédictions")
    st.markdown("Journal persistant de toutes les prédictions — manuelles et automatiques (Airflow).")

    log = fetch_predictions_log()

    if not log:
        st.info("Aucune prédiction enregistrée. Effectuez une prédiction ou lancez le DAG Airflow.")
    else:
        total   = len(log)
        fraudes = sum(1 for r in log if r.get("prediction") == 1)
        airflow_count = sum(1 for r in log if r.get("source") == "airflow")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total prédictions", total)
        c2.metric("🚨 Fraudes", fraudes)
        c3.metric("✅ Légitimes", total - fraudes)
        c4.metric("🤖 Via Airflow", airflow_count)

        st.divider()
        st.markdown('<p class="section-title">📄 Journal complet</p>', unsafe_allow_html=True)

        rows_trace = [
            {
                "Horodatage": r.get("timestamp", "—"),
                "Source": "🤖 Airflow" if r.get("source") == "airflow" else "👤 Manuel",
                "Résultat": "🚨 Fraude" if r.get("prediction") == 1 else "✅ Légitime",
                "Probabilité": f"{float(r.get('probability', 0)):.1%}",
            }
            for r in log
        ]
        df_trace = pd.DataFrame(rows_trace)
        st.dataframe(df_trace, use_container_width=True, hide_index=True)

        st.divider()
        csv_bytes = df_trace.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Exporter en CSV",
            data=csv_bytes,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.button("🔄 Rafraîchir", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
