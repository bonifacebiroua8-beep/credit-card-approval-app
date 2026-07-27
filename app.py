import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Évaluation de Crédit Carte",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Système d'Évaluation de Demande de Carte de Crédit")
st.write("Saisissez les caractéristiques du client pour obtenir une décision automatisée instantanée.")

# Chargement du modèle et du scaler
@st.cache_resource
def load_resources():
    model = joblib.load('best_model_credit_card.pkl')
    scaler = joblib.load('scaler_credit_card.pkl')
    return model, scaler

try:
    model, scaler = load_resources()
    st.sidebar.success("✅ Modèle et Scaler chargés")
except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers .pkl : {e}")

# Formulaire de saisie des données
st.subheader("📋 Formulaire du Client")

col1, col2, col3 = st.columns(3)

with col1:
    A1 = st.selectbox("A1 (Genre / Statut)", [0, 1])
    A2 = st.number_input("A2 (Âge)", min_value=15.0, max_value=90.0, value=30.0)
    A3 = st.number_input("A3 (Dette / Ratio)", min_value=0.0, max_value=30.0, value=5.0)
    A4 = st.selectbox("A4 (Statut marital)", [1, 2, 3])
    A5 = st.selectbox("A5 (Catégorie client)", list(range(1, 15)), index=3)

with col2:
    A6 = st.selectbox("A6 (Type d'emploi)", list(range(1, 10)), index=3)
    A7 = st.number_input("A7 (Ancienneté pro / années)", min_value=0.0, max_value=30.0, value=2.0)
    A8 = st.selectbox("A8 (Défaut de paiement passé)", [0, 1], help="0 = Non, 1 = Oui")
    A9 = st.selectbox("A9 (Emploi stable)", [0, 1], help="0 = Non, 1 = Oui")
    A10 = st.number_input("A10 (Score de crédit)", min_value=0, max_value=70, value=2)

with col3:
    A11 = st.selectbox("A11 (Permis de conduire)", [0, 1])
    A12 = st.selectbox("A12 (Type de citoyenneté)", [1, 2, 3])
    A13 = st.number_input("A13 (Code postal / Région)", min_value=0, max_value=2000, value=160)
    A14 = st.number_input("A14 (Revenu annuel / Épargne)", min_value=0, max_value=100000, value=1000)

# Bouton de prédiction
st.markdown("---")
if st.button("🔍 Analyser la demande", use_container_width=True):
    feature_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14']
    input_data = pd.DataFrame([[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14]], columns=feature_names)
    
    # Prétraitement et prédiction
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    proba = model.predict_proba(scaled_data)[0][1]
    
    # Affichage du résultat
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.success("### ✅ Décision : CRÉDIT APPROUVÉ")
        else:
            st.error("### ❌ Décision : CRÉDIT REJETÉ")
            
    with res_col2:
        st.metric(label="Probabilité d'approbation", value=f"{proba * 100:.2f}%")
        st.progress(float(proba))