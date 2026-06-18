"""DAG Airflow - trafic de previsions quotidien.

Seance 17 - TP Airflow (suite)
    Planifie l'envoi quotidien d'un lot de previsions a l'API : chaque jour a
    10h, on echantillonne 20 lignes du jeu de donnees et on les envoie en
    POST /predict. Cela simule un flux de previsions en production.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

N_PREDICTIONS = 20

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_send_predictions(**context) -> None:
    import httpx

    from ethereum_fraud.config import TARGET
    from ethereum_fraud.data import load_data

    api_url = os.environ.get("API_URL", "http://api:8000")

    features = load_data().drop(columns=[TARGET])
    sample = features.sample(n=N_PREDICTIONS, random_state=None)

    sent = 0
    with httpx.Client(base_url=api_url, timeout=10.0) as client:
        client.get("/health").raise_for_status()
        for _, row in sample.iterrows():
            payload = json.loads(row.to_json())
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            sent += 1

    logger.info("%d previsions envoyees a %s", sent, api_url)


with DAG(
    dag_id="daily_predictions",
    description="Envoie 20 previsions par jour a l'API (trafic simule)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 10 * * *",
    catchup=False,
    tags=["classification", "predictions"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
