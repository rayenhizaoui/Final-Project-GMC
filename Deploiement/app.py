from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from predictor import predict_insurance_cost

app = Flask(__name__)

# Charger le modèle et le scaler
MODEL_PATH = os.path.join('models', 'xgboost_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        data = request.get_json()
        
        age = int(data['age'])
        sex = data['sex']
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker = data['smoker']
        region = data['region']
        
        # Vérifier les contraintes des données
        if age < 18 or age > 100:
            return jsonify({'error': 'L\'âge doit être entre 18 et 100 ans'}), 400
            
        if bmi < 10 or bmi > 50:
            return jsonify({'error': 'L\'IMC doit être entre 10 et 50'}), 400
            
        if children < 0 or children > 10:
            return jsonify({'error': 'Le nombre d\'enfants doit être entre 0 et 10'}), 400
        
        # Faire la prédiction
        prediction = predict_insurance_cost(age, sex, bmi, children, smoker, region)
        
        # Formater la prédiction avec 2 décimales et le symbole $
        formatted_prediction = "${:,.2f}".format(prediction)
        
        return jsonify({
            'prediction': formatted_prediction,
            'raw_prediction': prediction
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)