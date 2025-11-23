import streamlit as st
import pandas as pd
import pickle
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


@st.cache_resource
def load_assets():
    df = pd.read_csv("data/January 2020 predictions.csv")

    with open("model_files/product_encoder.pkl", "rb") as f:
        product_encoder = pickle.load(f)

    with open("model_files/zone_encoder.pkl", "rb") as f:
        zone_encoder = pickle.load(f)

    return df, product_encoder, zone_encoder


def df_to_texts_daily(df):
    def row_to_text(row):
        product_label = row["Product"]
        zone_label = row["Geographic Zone"]

        return (
            f"Date: {row['OnlyDate']} | "
            f"Produit: {product_label} | "
            f"Zone Géographique: {zone_label} | "
            f"Quantité: {row['Quantity Ordered']}"
        )
    lines = [row_to_text(r) for _, r in df.iterrows()]
    return lines


def build_llm_chain(full_context):

    llm = Ollama(model="qwen2.5:14b")

    prompt = PromptTemplate(
    template="""
Tu es un expert en prévisions de la demande.
Tu dois répondre uniquement à partir du contexte fourni, qui représente des prédictions journalières.
Si l'information est insuffisante, dis-le clairement.

CONTEXT:
{context}

QUESTION:
{question}

La plupart des questions demanderont d'analyser les prédictions
Par exemple :
 - si l'on demande la zone géographique nécessitant une augmentation de stock → tu devras identifier la zone où la demande prévue est la plus élevée.
 - si l'on te demande une comparaison, analyse les quantités prévues sur les zones géographiques (ville) voir même les dates concernées.

Important :
 - Tu dois t'appuyer uniquement sur les prédictions fournies dans le CONTEXT.
 - N'invente jamais de chiffres ou de dates.
 - Commence TOUJOURS ta réponse en rappelant très brièvement les informations du contexte utiles à la question.

Réponse concise, structurée et utile :
""",
    input_variables=["context", "question"]
)
    # Le retriever fait planté la session pour une valeur de k supérieur à 5.
    # Pourtant, pour avoir de bon résultats, il faut récuperer au minimum 31 données pour un produit et une zone géographique donnée en contexe. On va donc mettre tout en contexte directement pour le test
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=prompt.template.replace(
                "CONTEXT:\n{context}", 
                f"CONTEXT:\n{full_context}"
            ),
            input_variables=["question"]
        )
    )

    return chain

def main():
    st.title("Assistant IA — Analyse des Prévisions Journalières")
    st.write("Posez une question basée sur les prédictions journalières générées par le LGBM.")

    df, product_encoder, zone_encoder = load_assets()

    texts = df_to_texts_daily(df, product_encoder, zone_encoder)

    full_context = "\n".join(texts)

    chain = build_llm_chain(full_context)

    question = st.text_input("Votre question sur les prévisions :")

    if question:
        with st.spinner("Analyse en cours..."):
            answer = chain.run({"question": question})

        st.markdown("### Réponse :")
        st.write(answer)

    with st.expander("Aperçu des données utilisées"):
        st.dataframe(df)


if __name__ == "__main__":
    main()
