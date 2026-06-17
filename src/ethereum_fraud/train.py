"""Entrainement du modele de classification (baseline).

Seance 5 - TP MLflow Tracking
    Entraine et evalue un modele logistique avec suivi MLflow complet.
    Lancement : python -m ethereum_fraud.train
"""

from __future__ import annotations

import argparse

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


from ethereum_fraud.config import MODEL_DIR
from ethereum_fraud.data import load_data, split
from ethereum_fraud.features import build_preprocessor
from ethereum_fraud.tracking import log_dataset, setup_experiment


def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )


def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    setup_experiment()

    with mlflow.start_run():
        log_dataset(df, context="training")

        mlflow.log_params({"c": c, "max_iter": max_iter})

        model = build_model(c=c, max_iter=max_iter)
        model.fit(x_train, y_train)

        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "f1": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
        print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["numpy.dtype"])

        # Matrice de confusion en artefact (S5-7 bonus)
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title("Matrice de confusion : LogisticRegression")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
