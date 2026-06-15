# Detection de fraude sur Ethereum — Pipeline MLOps

## Problematique

Les transactions sur la blockchain Ethereum sont publiques et tracables, mais cela n'empeche pas les activites malveillantes : arnaques, ponzi, phishing, blanchiment. Ce projet vise a **classifier automatiquement les adresses Ethereum comme frauduleuses (1) ou legitimes (0)** a partir de leur historique de transactions.

- **Les `1`** : adresses identifiees comme frauduleuses (arnaque confirmee, signalement Etherscan...)
- **Les `0`** : adresses legitimes sans historique frauduleux

Predire cette cible est utile pour alerter les utilisateurs avant un transfert vers une adresse a risque, ou pour alimenter des systemes de surveillance on-chain.

## Dataset

**Source** : [Ethereum Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)

| Caracteristique      | Valeur                          |
|---------------------|---------------------------------|
| Fichier             | `data/transaction_dataset.csv`  |
| Lignes              | 9 841 adresses                  |
| Colonnes            | 51 (dont cible + identifiants)  |
| Cible               | `FLAG` (0 = legitime, 1 = fraude) |
| Distribution        | 78 % legitimes / 22 % fraudes   |
| Features numeriques | 45 (volumes ETH, frequences, compteurs ERC20...) |
| Features categorielles | 2 (type de token ERC20 le plus envoye/recu) |

Les colonnes ERC20 contiennent des NaN pour les adresses sans activite ERC20 (~8 % des lignes) : ils sont imputes a 0 dans le pre-processing (valeur semantiquement correcte = pas d'activite).

## Mise en route

```bash
make install          # installe les dependances (uv)
```

Verifier que tout fonctionne :

```bash
python -m mlproject.train   # doit afficher f1=... roc_auc=...
```

## Structure du projet

```
Ethereum-fraud-detection/
  README.md                 ce fichier (journal de bord)
  Makefile                  cibles d'installation
  pyproject.toml            dependances du projet
  data/
    transaction_dataset.csv dataset Ethereum (Kaggle)
  mlproject/
    config.py               configuration du dataset        <- TP S0 (fait)
    data.py                 chargement + split
    features.py             pre-processing (imputation + scaling + OHE)
    evaluation.py           summary plot SHAP
    train.py                baseline MLflow                 <- TP S5
    train_optuna.py         optimisation Optuna + Registry  <- TP S6
    train_models.py         GridSearchCV + SHAP             <- TP S7
    api.py                  API FastAPI                     <- TP S12
  frontend/
    app.py                  frontend Streamlit              <- TP S14 bis
  docker/
    Dockerfile.train        image d'entrainement            <- TP S8
    Dockerfile.api          image de l'API
    Dockerfile.frontend     image du frontend
  docker-compose.yml        orchestration de la stack       <- TP S14
  dags/
    retrain_dag.py          DAG de re-entrainement Airflow  <- TP S17
```



### Seance 0 — Choix du dataset et configuration

- Dataset choisi : Ethereum Fraud Detection (Kaggle, 9 841 adresses, 51 colonnes)
- Problematique : classification binaire fraude / legitime sur la colonne `FLAG`
- Configuration de `mlproject/config.py` : TARGET, NUMERIC_FEATURES (45), CATEGORICAL_FEATURES (2)
- Ajout d'un `SimpleImputer` dans `features.py` pour gerer les NaN des colonnes ERC20
- Baseline logistique : **f1 = 0.971 / roc_auc = 0.998**
