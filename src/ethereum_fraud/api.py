"""API d'inference d'un modele de classification (FastAPI).

Seance 12 - TP FastAPI
    /health est fourni et fonctionne. A vous d'implementer le schema d'entree
    (adapte a VOTRE jeu de donnees), le schema de sortie, le chargement du
    modele et l'endpoint /predict (voir les TODO S12-n).
    Lancement : `uvicorn mlproject.api:app --reload`
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ethereum_fraud.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODEL_DIR / "model.joblib"
    if model_path.exists():
        ml["model"] = joblib.load(model_path)
        logger.info("Modèle chargé depuis %s", model_path)
    else:
        logger.warning("Modèle introuvable : %s. Lancez d'abord `make train`.", model_path)
    yield
    ml.clear()


app = FastAPI(title="Ethereum Fraud Detection API", version="1.0.0", lifespan=lifespan)


class Features(BaseModel):
    # Colonnes numériques
    avg_min_between_sent_tnx: Optional[float] = Field(None, alias="Avg min between sent tnx")
    avg_min_between_received_tnx: Optional[float] = Field(None, alias="Avg min between received tnx")
    time_diff_first_last: Optional[float] = Field(None, alias="Time Diff between first and last (Mins)")
    sent_tnx: Optional[float] = Field(None, alias="Sent tnx")
    received_tnx: Optional[float] = Field(None, alias="Received Tnx")
    number_of_created_contracts: Optional[float] = Field(None, alias="Number of Created Contracts")
    unique_received_from_addresses: Optional[float] = Field(None, alias="Unique Received From Addresses")
    unique_sent_to_addresses: Optional[float] = Field(None, alias="Unique Sent To Addresses")
    min_value_received: Optional[float] = Field(None, alias="min value received")
    max_value_received: Optional[float] = Field(None, alias="max value received ")
    avg_val_received: Optional[float] = Field(None, alias="avg val received")
    min_val_sent: Optional[float] = Field(None, alias="min val sent")
    max_val_sent: Optional[float] = Field(None, alias="max val sent")
    avg_val_sent: Optional[float] = Field(None, alias="avg val sent")
    min_value_sent_to_contract: Optional[float] = Field(None, alias="min value sent to contract")
    max_val_sent_to_contract: Optional[float] = Field(None, alias="max val sent to contract")
    avg_value_sent_to_contract: Optional[float] = Field(None, alias="avg value sent to contract")
    total_transactions: Optional[float] = Field(None, alias="total transactions (including tnx to create contract")
    total_ether_sent: Optional[float] = Field(None, alias="total Ether sent")
    total_ether_received: Optional[float] = Field(None, alias="total ether received")
    total_ether_sent_contracts: Optional[float] = Field(None, alias="total ether sent contracts")
    total_ether_balance: Optional[float] = Field(None, alias="total ether balance")
    total_erc20_tnxs: Optional[float] = Field(None, alias=" Total ERC20 tnxs")
    erc20_total_ether_received: Optional[float] = Field(None, alias=" ERC20 total Ether received")
    erc20_total_ether_sent: Optional[float] = Field(None, alias=" ERC20 total ether sent")
    erc20_total_ether_sent_contract: Optional[float] = Field(None, alias=" ERC20 total Ether sent contract")
    erc20_uniq_sent_addr: Optional[float] = Field(None, alias=" ERC20 uniq sent addr")
    erc20_uniq_rec_addr: Optional[float] = Field(None, alias=" ERC20 uniq rec addr")
    erc20_uniq_sent_addr_1: Optional[float] = Field(None, alias=" ERC20 uniq sent addr.1")
    erc20_uniq_rec_contract_addr: Optional[float] = Field(None, alias=" ERC20 uniq rec contract addr")
    erc20_avg_time_between_sent_tnx: Optional[float] = Field(None, alias=" ERC20 avg time between sent tnx")
    erc20_avg_time_between_rec_tnx: Optional[float] = Field(None, alias=" ERC20 avg time between rec tnx")
    erc20_avg_time_between_rec_2_tnx: Optional[float] = Field(None, alias=" ERC20 avg time between rec 2 tnx")
    erc20_avg_time_between_contract_tnx: Optional[float] = Field(None, alias=" ERC20 avg time between contract tnx")
    erc20_min_val_rec: Optional[float] = Field(None, alias=" ERC20 min val rec")
    erc20_max_val_rec: Optional[float] = Field(None, alias=" ERC20 max val rec")
    erc20_avg_val_rec: Optional[float] = Field(None, alias=" ERC20 avg val rec")
    erc20_min_val_sent: Optional[float] = Field(None, alias=" ERC20 min val sent")
    erc20_max_val_sent: Optional[float] = Field(None, alias=" ERC20 max val sent")
    erc20_avg_val_sent: Optional[float] = Field(None, alias=" ERC20 avg val sent")
    erc20_min_val_sent_contract: Optional[float] = Field(None, alias=" ERC20 min val sent contract")
    erc20_max_val_sent_contract: Optional[float] = Field(None, alias=" ERC20 max val sent contract")
    erc20_avg_val_sent_contract: Optional[float] = Field(None, alias=" ERC20 avg val sent contract")
    erc20_uniq_sent_token_name: Optional[float] = Field(None, alias=" ERC20 uniq sent token name")
    erc20_uniq_rec_token_name: Optional[float] = Field(None, alias=" ERC20 uniq rec token name")
    
    # Colonnes catégorielles
    erc20_most_sent_token_type: Optional[str] = Field(None, alias=" ERC20 most sent token type")
    erc20_most_rec_token_type: Optional[str] = Field(None, alias=" ERC20_most_rec_token_type")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "Avg min between sent tnx": 1200.5,
                    "Received Tnx": 10,
                    "total ether balance": 0.5,
                    " ERC20 most sent token type": "ERC20",
                    " ERC20_most_rec_token_type": "ERC20",
                }
            ]
        },
    }


class PredictionOut(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": "model" in ml}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    row = pd.DataFrame([features.model_dump(by_alias=True)])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


@app.get("/model-info")
def model_info() -> dict:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}