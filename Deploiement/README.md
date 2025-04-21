# Prédicteur de Coûts d'Assurance Santé

Application web basée sur Flask et XGBoost pour prédire les coûts d'assurance santé en fonction des caractéristiques des clients.

## Fonctionnalités

- Formulaire interactif pour saisir les informations du client
- Calcul d'IMC intégré
- Prédiction du coût d'assurance en temps réel
- Interface utilisateur responsive et moderne

## Architecture technique

- **Backend** : Flask (Python)
- **Modèle ML** : XGBoost
- **Frontend** : HTML/CSS/JavaScript
- **Données** : Jeu de données CSV (insurance.csv)

## Installation

1. Cloner le dépôt
2. Installer les dépendances :
   ```
   pip install -r requirements.txt
   ```
3. Entraîner le modèle (nécessaire lors de la première installation) :
   ```
   python train_model.py
   ```
4. Lancer l'application :
   ```
   python app.py
   ```
5. Accéder à l'application via http://localhost:5000

## Entraînement du modèle

Le modèle est entraîné à partir du fichier CSV situé dans `data/insurance.csv`. Le script `train_model.py` :

- Prétraite les données
- Crée des fonctionnalités avancées
- Entraîne un modèle XGBoost
- Sauvegarde le modèle et le scaler dans le dossier `models/`

## Structure du projet

- `app.py` : Application Flask principale
- `predictor.py` : Fonctions pour effectuer les prédictions
- `train_model.py` : Script pour entraîner et sauvegarder le modèle
- `data/` : Dossier contenant les données d'entraînement
- `models/` : Dossier contenant les modèles sauvegardés
- `templates/` : Templates HTML pour l'interface utilisateur
- `static/` : Fichiers statiques (CSS, JavaScript, images)

## Déploiement

Ce projet peut être déployé sur n'importe quelle plateforme supportant Python et Flask :

1. Pour le déploiement en production, utilisez :
   ```
   gunicorn app:app
   ```
2. Configurez les variables d'environnement appropriées pour la production

## Auteur

Rayen Hizaoui

