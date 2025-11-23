# 📘 Demand Forecasting Assistant  
### (LLM + Streamlit + Ollama + LightGBM + RAG)

Ce projet fournit une interface Streamlit permettant d’interagir avec un **assistant intelligent spécialisé en prévisions de la demande**, basé sur :

- Un modèle **LGBM** déjà entraîné pour prédire la demande **journalière**
- Les LabelEncoder associés
- Un csv contenant les prédictions pour les 31 jours après les données initiales
- Un pipeline **RAG** utilisant SentenceTransformer + ChromaDB.
- Un grand modèle de langage **Qwen2.5:14B** exécuté via **Ollama**.
- Une interface Web pour poser des questions et obtenir des réponses analytiques.
- Une architecture **Docker Compose** comprenant un service Ollama et un service Streamlit.
  
Le notebook **Quantity Order Forecast.ipynb** correspond aux étapes suivis pour l'entraînement du modèle de prédictions.
---

## 🧩 Prérequis

- Docker  
- Docker Compose  
- (Optionnel mais conseillé) au moins 16 Go de RAM pour Qwen2.5:14B
---

## 🚀 Installation & Lancement

Lancer simplement depuis la racine du projet :

```bash
docker-compose up --build
```
Ce qui va :

Démarrer Ollama

Télécharger automatiquement qwen2.5:14b et servir le modèle

Démarrer Streamlit

Servir l’interface sur :

👉 http://localhost:8501
