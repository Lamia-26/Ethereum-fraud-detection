"""DAG Airflow - pipeline de re-entrainement du modele.

Seance 17 - TP Airflow
    Pipeline simple : preparation des donnees -> entrainement -> controle
    qualite. Se lance tous les lundis a 3h du matin.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    from ethereum_fraud.data import load_data

    df = load_data()
    logger.info("Donnees OK : %d lignes, %d colonnes", len(df), df.shape[1])


def task_train(**context) -> None:
    from ethereum_fraud.train import train

    metrics = train()
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    logger.info("Entrainement termine : F1=%.4f  ROC-AUC=%.4f", metrics["f1"], metrics["roc_auc"])


def task_check_quality(**context) -> None:
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 is None or f1 < QUALITY_THRESHOLD:
        raise ValueError(
            f"Qualite insuffisante : F1={f1} < seuil={QUALITY_THRESHOLD}"
        )
    logger.info("Qualite validee : F1=%.4f >= %.2f", f1, QUALITY_THRESHOLD)


with DAG(
    dag_id="model_retraining",
    description="Prepare les donnees, reentraine le modele et controle sa qualite",
    schedule="0 3 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training"],
) as dag:
    prepare     = PythonOperator(task_id="prepare_data",   python_callable=task_prepare_data)
    train_task  = PythonOperator(task_id="train",          python_callable=task_train)
    check       = PythonOperator(task_id="check_quality",  python_callable=task_check_quality)

    prepare >> train_task >> check
