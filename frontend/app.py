"""Frontend Streamlit : detecteur de fraude Ethereum."""
from __future__ import annotations

import os
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st

from ethereum_fraud.config import (
    CATEGORICAL_FEATURES,
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    NUMERIC_FEATURES,
)
from ethereum_fraud.tracking import get_all_runs, get_latest_confusion_matrix, get_latest_metrics, get_latest_shap

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
API_EXTERNAL_URL = os.environ.get("API_EXTERNAL_URL", API_URL)
MLFLOW_EXTERNAL_URL = os.environ.get("MLFLOW_EXTERNAL_URL", "http://localhost:5000")

st.set_page_config(page_title="Ethereum Fraud Detection", layout="wide")

st.markdown("""
<style>
/* Masquer la barre d'outils Streamlit */
header[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

/* Header hero */
.hero {
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #0EA5E9 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 { color: white; margin: 0 0 0.5rem 0; font-size: 2rem; }
.hero p  { color: #BFDBFE; margin: 0; font-size: 1rem; }

/* Cartes metriques colorees */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A8A 0%, #1E40AF 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] [data-testid="metric-container"] {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
}

/* Boutons sidebar */
[data-testid="stSidebar"] [data-testid="stLinkButton"] a {
    background: rgba(255,255,255,0.15) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.4) !important;
    border-radius: 8px;
    font-weight: 600;
}
[data-testid="stSidebar"] [data-testid="stLinkButton"] a:hover {
    background: rgba(255,255,255,0.25) !important;
}

/* Boutons */
[data-testid="stButton"] button {
    border-radius: 8px;
    font-weight: 600;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# BENHADDAD Lamia")
    st.divider()
    st.markdown("## Navigation")
    health = fetch_health()
    model_ready = health.get("model_ready", False)

    if model_ready:
        st.success("API operationnelle")
    else:
        st.error("API indisponible")

    metrics = cached_metrics()
    if metrics:
        st.metric("F1", f"{metrics.get('f1', 0):.3f}")
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

    st.divider()
    st.link_button("Ouvrir MLflow", MLFLOW_EXTERNAL_URL, use_container_width=True)

# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------
tab_accueil, tab_predict, tab_experiments, tab_monitoring = st.tabs([
    "Accueil", "Prediction", "Experiences", "Monitoring"
])

# ===========================================================================
# ACCUEIL
# ===========================================================================
with tab_accueil:
    st.markdown("""
    <div class="hero">
        <h1>Detection de fraude sur Ethereum</h1>
        <p>Pipeline MLOps complet pour detecter automatiquement les transactions frauduleuses
        sur la blockchain Ethereum a partir de caracteristiques comportementales des portefeuilles.</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Features numeriques", len(NUMERIC_FEATURES))
    col2.metric("Features categorielles", len(CATEGORICAL_FEATURES))
    col3.metric("Total features", len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))
    col4.metric("Cible", "FLAG (0/1)")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Modele en production")
        info = fetch_model_info()
        st.markdown(f"- **Algorithme** : Logistic Regression")
        st.markdown(f"- **Version** : {info.get('version', 'inconnue')}")
        if metrics:
            st.markdown(f"- **F1** : {metrics.get('f1', 0):.3f}")
            st.markdown(f"- **ROC-AUC** : {metrics.get('roc_auc', 0):.3f}")
        st.markdown(f"- **Seuil F1 minimum** : {EVAL_F1_MIN}")
        st.markdown(f"- **Seuil ROC-AUC minimum** : {EVAL_ROC_AUC_MIN}")

    with col_b:
        st.subheader("Stack technique")
        st.markdown("""
        - **Tracking** : MLflow
        - **API** : FastAPI + Uvicorn
        - **Frontend** : Streamlit
        - **Conteneurisation** : Docker Compose
        - **CI/CD** : GitHub Actions
        """)

    st.divider()
    with st.expander("Voir toutes les features numeriques"):
        cols = st.columns(2)
        for i, feat in enumerate(NUMERIC_FEATURES):
            cols[i % 2].markdown(f"- `{feat}`")

    with st.expander("Voir les features categorielles"):
        for feat in CATEGORICAL_FEATURES:
            st.markdown(f"- `{feat}`")

# ===========================================================================
# PREDICTION
# ===========================================================================
with tab_predict:
    st.title("Tester une transaction")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Transactions**")
            sent_tnx = st.number_input("Sent tnx", min_value=0.0, value=0.0)
            received_tnx = st.number_input("Received Tnx", min_value=0.0, value=0.0)
            avg_min_sent = st.number_input("Avg min between sent tnx", min_value=0.0, value=0.0)
            avg_min_received = st.number_input("Avg min between received tnx", min_value=0.0, value=0.0)
            time_diff = st.number_input("Time Diff between first and last (Mins)", min_value=0.0, value=0.0)
            nb_contracts = st.number_input("Number of Created Contracts", min_value=0.0, value=0.0)

        with col2:
            st.markdown("**Ether**")
            total_sent = st.number_input("total Ether sent", min_value=0.0, value=0.0)
            total_received = st.number_input("total ether received", min_value=0.0, value=0.0)
            total_balance = st.number_input("total ether balance", value=0.0)
            total_sent_contracts = st.number_input("total ether sent contracts", min_value=0.0, value=0.0)

        st.markdown("**ERC20**")
        col3, col4 = st.columns(2)
        with col3:
            erc20_most_sent = st.selectbox("ERC20 most sent token type",
                                           ["", "ERC20", "ERC721", "ERC1155", "other"])
        with col4:
            erc20_most_rec = st.selectbox("ERC20 most rec token type",
                                          ["", "ERC20", "ERC721", "ERC1155", "other"])

        submitted = st.form_submit_button("Predire", use_container_width=True)

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
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
            st.session_state.history.insert(0, st.session_state.last_result)
        except httpx.HTTPError as exc:
            st.error(f"Appel a l'API impossible : {exc}")
            st.session_state.last_result = None

    if st.session_state.last_result:
        res = st.session_state.last_result
        prediction = res["prediction"]
        probability = res["probability"]

        st.divider()
        if prediction == 1:
            st.error("### Transaction frauduleuse detectee")
        else:
            st.success("### Transaction legitime")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Resultat", "Fraude" if prediction == 1 else "Legitime")
        col_b.metric("Probabilite de fraude", f"{probability:.1%}")
        col_c.metric("Heure", res["timestamp"])
        st.progress(probability, text=f"Score de risque : {probability:.1%}")

        st.divider()
        st.markdown("**Cette prediction est-elle correcte ?**")
        col_yes, col_no = st.columns(2)
        if col_yes.button("Correcte", use_container_width=True):
            st.session_state.history[0]["feedback"] = "correct"
            st.success("Merci pour votre retour !")
        if col_no.button("Incorrecte", use_container_width=True):
            st.session_state.history[0]["feedback"] = "incorrect"
            st.warning("Merci, nous en tiendrons compte.")

        st.divider()
        st.subheader("Historique de la session")
        if st.session_state.history:
            rows = [
                {
                    "Heure": h["timestamp"],
                    "Prediction": "Fraude" if h["prediction"] == 1 else "Legitime",
                    "Probabilite": f"{h['probability']:.1%}",
                    "Feedback": h.get("feedback") or "-",
                }
                for h in st.session_state.history
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            if st.button("Effacer"):
                st.session_state.history = []
                st.session_state.last_result = None
                st.rerun()

# ===========================================================================
# EXPERIENCES MLFLOW
# ===========================================================================
with tab_experiments:
    st.title("Historique des experiences MLflow")

    runs = cached_runs()
    if not runs:
        st.info("Aucune experience trouvee. Lancez un entrainement d'abord.")
    else:
        st.metric("Nombre de runs", len(runs))
        st.divider()

        rows = []
        for r in runs:
            rows.append({
                "Date": r["date"],
                "Nom": r["name"],
                "Statut": r["status"],
                "F1": round(r["metrics"].get("f1", 0), 4),
                "ROC-AUC": round(r["metrics"].get("roc_auc", 0), 4),
                **{f"param:{k}": v for k, v in r["params"].items()},
            })
        df_runs = pd.DataFrame(rows)
        st.dataframe(df_runs, use_container_width=True)

        st.divider()
        st.subheader("Evolution des metriques")
        df_chart = pd.DataFrame([
            {"Run": r["name"], "F1": r["metrics"].get("f1", 0),
             "ROC-AUC": r["metrics"].get("roc_auc", 0)}
            for r in reversed(runs)
        ]).set_index("Run")
        st.line_chart(df_chart, use_container_width=True)

        st.divider()
        best = max(runs, key=lambda r: r["metrics"].get("roc_auc", 0))
        st.subheader("Meilleur run")
        col1, col2, col3 = st.columns(3)
        col1.metric("Nom", best["name"])
        col2.metric("F1", f"{best['metrics'].get('f1', 0):.4f}")
        col3.metric("ROC-AUC", f"{best['metrics'].get('roc_auc', 0):.4f}")

        st.divider()
        st.subheader("Matrice de confusion (dernier run)")
        cm_bytes = cached_confusion_matrix()
        if cm_bytes:
            st.image(cm_bytes, width=450)
        else:
            st.info("Matrice de confusion non disponible.")

# ===========================================================================
# MONITORING
# ===========================================================================
with tab_monitoring:
    st.title("Monitoring du modele")

    metrics = cached_metrics()
    health = fetch_health()
    model_ready = health.get("model_ready", False)

    # Statut general
    if model_ready and metrics:
        f1_ok = metrics.get("f1", 0) >= EVAL_F1_MIN
        roc_ok = metrics.get("roc_auc", 0) >= EVAL_ROC_AUC_MIN
        if f1_ok and roc_ok:
            st.success("Le modele est en production et respecte les seuils de qualite.")
        else:
            st.warning("Le modele est en production mais certains seuils ne sont pas atteints.")
    else:
        st.error("Le modele n'est pas disponible.")

    st.divider()
    st.subheader("Metriques du modele en production")

    if metrics:
        col1, col2 = st.columns(2)

        f1 = metrics.get("f1", 0)
        roc = metrics.get("roc_auc", 0)

        with col1:
            st.markdown("**F1 Score**")
            st.metric(
                label="F1",
                value=f"{f1:.4f}",
                delta=f"{f1 - EVAL_F1_MIN:+.4f} vs seuil ({EVAL_F1_MIN})",
                delta_color="normal",
            )
            st.progress(f1)

        with col2:
            st.markdown("**ROC-AUC**")
            st.metric(
                label="ROC-AUC",
                value=f"{roc:.4f}",
                delta=f"{roc - EVAL_ROC_AUC_MIN:+.4f} vs seuil ({EVAL_ROC_AUC_MIN})",
                delta_color="normal",
            )
            st.progress(roc)
    else:
        st.info("Metriques MLflow inaccessibles.")

    st.divider()
    st.subheader("Statut de l'API")
    col1, col2 = st.columns(2)
    col1.metric("Modele charge", "Oui" if model_ready else "Non")
    with col2:
        st.link_button("Ouvrir l'API (docs)", f"{API_EXTERNAL_URL}/docs", use_container_width=True)

    st.divider()
    st.subheader("Statistiques de la session")
    total = len(st.session_state.history)
    fraudes = sum(1 for h in st.session_state.history if h["prediction"] == 1)
    col1, col2, col3 = st.columns(3)
    col1.metric("Predictions totales", total)
    col2.metric("Fraudes detectees", fraudes)
    col3.metric("Legitimes", total - fraudes)

    st.divider()
    st.subheader("Artefacts du dernier run")
    col_cm, col_shap = st.columns(2)
    with col_cm:
        st.markdown("**Matrice de confusion**")
        cm_bytes = cached_confusion_matrix()
        if cm_bytes:
            st.image(cm_bytes)
        else:
            st.info("Non disponible.")
    with col_shap:
        st.markdown("**Importance des variables (SHAP)**")
        shap_bytes = cached_shap()
        if shap_bytes:
            st.image(shap_bytes)
        else:
            st.info("Non disponible — lancez `make train-models` pour generer le SHAP.")

    st.divider()
    st.link_button("Ouvrir MLflow UI", MLFLOW_EXTERNAL_URL, use_container_width=True)
