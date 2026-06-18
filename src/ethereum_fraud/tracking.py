"""Configuration partagee du suivi MLflow.

Centralise la configuration du tracking pour eviter de la dupliquer dans
chaque script d'entrainement, et ajoute la tracabilite des donnees
(dataset lineage).

Les scripts (train, train_models, train_optuna, evaluate) appellent
`setup_experiment()` au lieu de repeter `set_tracking_uri` + `set_experiment`.
"""

from __future__ import annotations

import logging
from datetime import datetime

import mlflow
import mlflow.data
import pandas as pd

from ethereum_fraud.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    TARGET,
)

logger = logging.getLogger(__name__)


def setup_experiment() -> None:
    """Configurer le tracking MLflow et les metadonnees de l'experience.

    Idempotent : re-appelable sans erreur.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    client = mlflow.MlflowClient()
    if MLFLOW_EXPERIMENT_DESCRIPTION:
        client.set_experiment_tag(
            experiment.experiment_id,
            "mlflow.note.content",
            MLFLOW_EXPERIMENT_DESCRIPTION,
        )
    for key, value in MLFLOW_EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment.experiment_id, key, value)
    logger.info("MLflow : %s  (experience : %s)", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT)


def get_latest_metrics() -> dict[str, float]:
    """Retourne les metriques du dernier run MLflow de l'experience."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        return {}
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return {}
    return {k: float(v) for k, v in runs[0].data.metrics.items()}


def get_all_runs() -> list[dict]:
    """Retourne tous les runs MLflow de l'experience, du plus recent au plus ancien."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        return []
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    result = []
    for run in runs:
        result.append({
            "run_id": run.info.run_id,
            "name": run.info.run_name or run.info.run_id[:8],
            "status": run.info.status,
            "date": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
            "metrics": {k: float(v) for k, v in run.data.metrics.items()},
            "params": run.data.params,
        })
    return result


def get_latest_confusion_matrix() -> bytes | None:
    """Telecharge et retourne la matrice de confusion du dernier run en bytes."""
    import tempfile

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    run_id = runs[0].info.run_id
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = client.download_artifacts(run_id, "confusion_matrix.png", tmp)
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    """Logger un dataset MLflow dans le run courant (tracabilite donnees -> modele).

    Parameters
    ----------
    df : pandas.DataFrame
        Donnees a referencer (features + cible).
    context : str
        Role du dataset dans le run, par exemple "training" ou "evaluation".
    name : str, optional
        Nom logique du dataset, par defaut "dataset".
    """
    dataset = mlflow.data.from_pandas(df, source=str(DATA_PATH), targets=TARGET, name=name)  # type: ignore[attr-defined]
    mlflow.log_input(dataset, context=context)
