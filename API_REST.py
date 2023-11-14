# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:19:50 2023

@author: laura.chacon
"""
from flask import Flask, request, jsonify
import pickle

import numpy as np
import pandas as pd
import json
import shap

app = Flask(__name__)

def drop_columns(X):
    return X.drop(["customerID","Churn"], axis=1)


# Cargar los pipelines con pickle
with open('models/knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

with open('models/logreg_model.pkl', 'rb') as file:
    logreg_model = pickle.load(file)

with open('models/svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('models/rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)
    
@app.route('/<model_version>/predict', methods=['POST'])
def predict(model_version):
    try:
        # Obtener datos de entrada desde la solicitud POST
        datos = request.get_json()
        #Cargue de datos
        churn_df = pd.DataFrame(datos)
        churn_df['TotalCharges'] = pd.to_numeric(churn_df['TotalCharges'], errors='raise')
        # Realizar la predicción utilizando el modelo1
        if model_version == 'knn_model':
            modelo = knn_model
        elif model_version == 'logreg_model': 
            modelo = logreg_model
        elif model_version == 'svm_model': 
            modelo = svm_model
        elif model_version == 'rf_model': 
            modelo = rf_model
        else:
            return jsonify({"error":"modelo no soportado"})
        resultado = modelo.predict_proba(churn_df)
        # Concatenar el ndarray al DataFrame existente
        labels = np.argmax(resultado, axis=1)
        resultado_df = pd.concat([pd.DataFrame(labels , columns=["label"]), pd.DataFrame(resultado, columns=['prob_no','prob_si'])], axis=1)
        resultado_df['label'] = resultado_df['label'].replace({0: 'No', 1: 'Yes'})
        return jsonify(json.loads(resultado_df.to_json(orient='records')))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/<model_version>/explain', methods=['POST'])
def explain(model_version):
    try:
       # Obtener datos de entrada desde la solicitud POST
        datos = request.get_json()
        #Cargue de datos
        churn_df = pd.DataFrame(datos)
        churn_df['TotalCharges'] = pd.to_numeric(churn_df['TotalCharges'], errors='raise')
        churn_df = drop_columns(churn_df)
        # Realizar la predicción utilizando el modelo1
        if model_version == 'knn_model':
            modelo = knn_model
            classifier = 'classifier'
        elif model_version == 'logreg_model': 
            modelo = logreg_model
            classifier = 'logreg'
        elif model_version == 'svm_model': 
            classifier = 'svm'
            modelo = svm_model
        elif model_version == 'rf_model': 
            modelo = rf_model
            classifier = 'rf'
        else:
            return jsonify({"error":"modelo no soportado"})
        X_t = pd.DataFrame(
            modelo["encoder"].fit_transform(churn_df),
            columns=[f.split("__")[1] for f in modelo["encoder"].get_feature_names_out()])
        def model(X):
            return modelo[classifier].predict_proba(X)[:,1]
        explainer = shap.Explainer(model, X_t)
        shap_values = explainer(X_t)
        columns=X_t.columns.tolist()
        resultado = []        
        for value in shap_values.values:
            valores_absolutos = np.abs(value)
            indices_maximos = np.argsort(valores_absolutos)[-3:]
            resultado.append({"feature1": columns[indices_maximos[-1]],"feature2": columns[indices_maximos[-2]],"feature3": columns[indices_maximos[-3]]})
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == '__main__':
    app.run(debug=False)