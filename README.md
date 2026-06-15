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


