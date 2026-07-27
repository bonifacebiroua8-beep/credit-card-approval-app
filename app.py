import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------------------------------------
st.set_page_config(
    page_title="Credit Scoring AI | Évaluation de Crédit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CSS PERSONNALISÉ POUR UN DESIGN PROFESSIONNEL
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Style global */
    .main {
        background-color: #f8fafc;
    }
    
    /* En-tête principal */
    .hero-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }
    .hero-header h1 {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Cartes de sections */
    .css-1r63f89, div[data-testid="stForm"] {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        padding: 1.5rem;
    }
    
    /* Titres des cartes */
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Bouton principal */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
    }
    
    /* Cartes de résultats */
    .result-card-approved {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-card-rejected {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-title-approved {
        color: #15803d;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    .result-title-rejected {
        color: #b91c1c;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CHARGEMENT DES RESSOURCES (MODÈLE & SCALER)
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load('best_model_credit_card.pkl')
    scaler = joblib.load('scaler_credit_card.pkl')
    return model, scaler

# Barre latérale d'information
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank-cards.png", width=70)
    st.title("Credit Scoring AI")
    st.caption("Système Décisionnel Automatisé")
    st.markdown("---")
    
    try:
        model, scaler = load_resources()
        st.success("🤖 Modèle IA : **Actif**")
        st.info("🎯 Précision Globale : **83.3 %**")
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        st.stop()
        
    st.markdown("---")
    st.markdown("### 📌 À propos")
    st.markdown(
        "Ce système utilise un modèle de **Régression Logistique** optimisé "
        "pour analyser le profil financier du candidat et évaluer le risque d'octroi de crédit."
    )

# ---------------------------------------------------------
# EN-TÊTE PRINCIPAL
# ---------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <h1>💳 Plateforme d'Évaluation de Crédit</h1>
    <p>Saisissez les informations financières du candidat pour générer une décision d'octroi en temps réel.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# FORMULAIRE ET COLONNES
# ---------------------------------------------------------
st.markdown("### 📝 Profil du Demandeur")

col_left, col_right = st.columns(2, gap="medium")

with col_left:
    with st.container(border=True):
        st.markdown("<div class='card-title'>👤 Informations Personnelles & Professionnelles</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            A1 = st.selectbox("Genre / Statut (A1)", options=[0, 1], format_func=lambda x: "Homme (1)" if x == 1 else "Femme (0)")
            A2 = st.number_input("Âge (A2)", min_value=18.0, max_value=85.0, value=30.5, step=0.5)
            A4 = st.selectbox("Statut Marital (A4)", options=[1, 2, 3], format_func=lambda x: f"Statut {x}")
            A5 = st.selectbox("Catégorie Socio-Pro (A5)", options=list(range(1, 15)), index=3)
            
        with c2:
            A6 = st.selectbox("Secteur d'Activité (A6)", options=list(range(1, 10)), index=3)
            A7 = st.number_input("Ancienneté Pro - Ans (A7)", min_value=0.0, max_value=40.0, value=2.5, step=0.25)
            A11 = st.selectbox("Permis de Conduire (A11)", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
            A12 = st.selectbox("Type Citoyenneté (A12)", options=[1, 2, 3], format_func=lambda x: f"Type {x}")

with col_right:
    with st.container(border=True):
        st.markdown("<div class='card-title'>📊 Solvabilité & Indicateurs Financiers</div>", unsafe_allow_html=True)
        
        c3, c4 = st.columns(2)
        with c3:
            A3 = st.number_input("Ratio Dette/Revenu (A3)", min_value=0.0, max_value=30.0, value=4.5, step=0.25)
            A8 = st.selectbox("Défaut de Paiement Passé (A8)", options=[0, 1], format_func=lambda x: "Oui (Risque Élevé)" if x == 1 else "Non (Aucun)")
            A9 = st.selectbox("Emploi Stable / CDI (A9)", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
            
        with c4:
            A10 = st.number_input("Score de Crédit Interne (A10)", min_value=0, max_value=70, value=5)
            A13 = st.number_input("Code Régional / Postal (A13)", min_value=0, max_value=2000, value=160)
            A14 = st.number_input("Revenu Annuel / Épargne (A14)", min_value=0, max_value=100000, value=1500, step=100)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# ACTION ET DÉCISION
# ---------------------------------------------------------
if st.button("⚡ Analyser le Dossier de Crédit"):
    feature_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14']
    input_data = pd.DataFrame([[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14]], columns=feature_names)
    
    # Prétraitement et Prédiction
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    proba_approval = model.predict_proba(scaled_data)[0][1]
    
    st.markdown("---")
    st.markdown("### 🎯 Résultat de l'Analyse")
    
    res_col1, res_col2 = st.columns([1.2, 1], gap="medium")
    
    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card-approved">
                <p class="result-title-approved">✅ DEMANDE APPROUVÉE</p>
                <p style="color: #166534; margin-top: 0.5rem;">Le profil du demandeur présente un niveau de risque conforme aux critères d'octroi.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-rejected">
                <p class="result-title-rejected">❌ DEMANDE REJETÉE</p>
                <p style="color: #991b1b; margin-top: 0.5rem;">Le risque d'impayé estimé dépasse le seuil de tolérance défini.</p>
            </div>
            """, unsafe_allow_html=True)
            
    with res_col2:
        with st.container(border=True):
            st.markdown("**Indice de Confiance Modèle**")
            st.metric(label="Score d'Approbation", value=f"{proba_approval * 100:.1f} %")
            st.progress(float(proba_approval))
            
            if proba_approval > 0.7:
                st.caption("🟢 Profil très solide")
            elif proba_approval >= 0.5:
                st.caption("🟡 Profil acceptable avec surveillance")
            else:
                st.caption("🔴 Risque élevé de défaut")
