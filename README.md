# 💳 Credit Card Approval Prediction App

Une application web interactive de Machine Learning développée avec **Streamlit**, conçue pour prédire l'approbation ou le rejet d'une demande de carte de crédit en fonction du profil financier et démographique du demandeur.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Deployed-success)

---

## 📌 Présentation du Projet

Dans le secteur bancaire et financier, l'évaluation des risques de crédit est un processus critique. Cette application automatise et accélère la prise de décision en s'appuyant sur un modèle de classification supervisée entraîné pour analyser les risques associés à un dossier de demande de crédit.

L'utilisateur peut renseigner interactivement différents critères (revenus, antécédents de crédit, dettes, situation professionnelle, etc.) et obtenir une **prédiction instantanée** sur l'éligibilité de la carte de crédit, accompagnée d'un indice de confiance.

---

## 🚀 Fonctionnalités Clés

* **Interface Utilisateur Intuitive :** Un formulaire dynamique sur Streamlit permettant de saisir facilement les informations du demandeur.
* **Modèle Prédictif Entraîné :** Utilisation d'un algorithme de Machine Learning (Classification binaire) optimisé pour minimiser les faux positifs (accorder un crédit à risque) et faux négatifs.
* **Évaluation en Temps Réel :** Calcul instantané de la décision (`Approuvé` / `Refusé`) dès la soumission du formulaire.
* **Hébergement Cloud :** Application entièrement déployée et accessible en ligne via Streamlit Community Cloud.

---

## 🛠️ Stack Technique

* **Langage :** Python 3.10+
* **Interface Web :** Streamlit
* **Machine Learning / Prétraitement :** Scikit-Learn, Pandas, NumPy
* **Sérialisation du modèle :** Joblib / Pickle (`.pkl`)

---

## 📁 Structure du Dépôt

```text
├── app.py                   # Code principal de l'application Streamlit
├── requirements.txt         # Dépendances et bibliothèques requises
├── best_model_credit_card.pkl    # Modèle de Machine Learning entraîné
│              
└── README.md                # Documentation du projet
