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

QUALITY_THRESHOLD = 0.80

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
    from ethereum_fraud.train_optuna import optimize

    results = optimize(n_trials=20, cv=3)
    best = results[0]
    context["ti"].xcom_push(key="roc_auc", value=best.test_roc_auc)
    logger.info("Meilleur modele : %s  test_roc_auc=%.4f", best.spec.name, best.test_roc_auc)


def task_check_quality(**context) -> None:
    roc_auc = context["ti"].xcom_pull(task_ids="train", key="roc_auc")
    if roc_auc is None or roc_auc < QUALITY_THRESHOLD:
        raise ValueError(
            f"Qualite insuffisante : roc_auc={roc_auc} < seuil={QUALITY_THRESHOLD}"
        )
    logger.info("Qualite validee : roc_auc=%.4f >= %.2f", roc_auc, QUALITY_THRESHOLD)


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
