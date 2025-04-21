import pandas as pd
import numpy as np
import joblib
import os

# Chemins vers les fichiers de modèle
MODEL_PATH = os.path.join('models', 'xgboost_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

# Charger le modèle et le scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Liste des colonnes numériques pour le scaling
num_cols = ['age', 'bmi', 'children', 'age_squared', 'bmi_squared', 'smoker_bmi', 'smoker_age']

def bmi_category(bmi):
    """Catégoriser l'IMC selon les standards médicaux"""
    if bmi < 18.5: return 0  # Sous-poids
    elif bmi < 25: return 1  # Normal
    elif bmi < 30: return 2  # Surpoids
    else: return 3  # Obésité

def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    """
    Prédit le coût d'assurance pour un nouveau client.
    
    Paramètres:
    - age: âge du client (int)
    - sex: sexe du client ('male' ou 'female')
    - bmi: indice de masse corporelle (float)
    - children: nombre d'enfants couverts (int)
    - smoker: si le client est fumeur ('yes' ou 'no')
    - region: région du client ('northeast', 'northwest', 'southeast', 'southwest')
    
    Retourne:
    - Le coût d'assurance prédit
    """
    # Convertir en valeurs numériques
    sex_val = 1 if sex == 'female' else 0
    smoker_val = 1 if smoker == 'yes' else 0
    
    # Créer un dictionnaire pour les régions
    regions = {
        'northeast': [1, 0, 0, 0],
        'northwest': [0, 1, 0, 0],
        'southeast': [0, 0, 1, 0],
        'southwest': [0, 0, 0, 1]
    }
    region_vals = regions.get(region, [0, 0, 0, 0])  # Default en cas de région inconnue
    
    # Créer un DataFrame pour le nouveau client
    new_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_val],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_val],
        'region_northeast': [region_vals[0]],
        'region_northwest': [region_vals[1]],
        'region_southeast': [region_vals[2]],
        'region_southwest': [region_vals[3]]
    })
    
    # Ajouter les features engineered
    new_data['age_squared'] = new_data['age'] ** 2
    new_data['bmi_squared'] = new_data['bmi'] ** 2
    new_data['bmi_category'] = new_data['bmi'].apply(bmi_category)
    new_data['smoker_bmi'] = new_data['smoker'] * new_data['bmi']
    new_data['smoker_age'] = new_data['smoker'] * new_data['age']
    
    # Scaling
    new_data[num_cols] = scaler.transform(new_data[num_cols])
    
    # Prédire
    prediction = model.predict(new_data)[0]
    
    # Convertir le résultat numpy.float32 en float Python standard
    prediction_float = float(prediction)
    
    return prediction_float