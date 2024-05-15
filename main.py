import streamlit as st
from src.ner import NaturalEntityRecognizer
from src.wiki import search_wikipedia
from src.vdb import WikiVectorDatabase
from src.groq import get_groq_completions


PROMPT_TEMPLATE = """Tu es un journaliste senior qui aime rétablir la vérité dans les informations. Répond à la 
question en français et dit que tu ne sais pas si tu n'as pas l'information. Tu peux t'aider du contexte suivant, 
qui provient directement de Wikipedia, pour appuyer tes propos :

{context}

Question :

{question}
"""


# FONCTIONS UTILISÉES LORS DU CLIC SUR LES BOUTONS DE L'INTERFACE
def on_button_click():
    st.session_state.run = True


def on_reset():
    st.session_state.run = False


# INITIALISATION DES PARAMÈTRES DU PROJET
ner = NaturalEntityRecognizer('Clemnt73/RoBERTa-ner')
embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"
vdb_creator = WikiVectorDatabase(
    embeddings_model_name=embeddings_model_name
)


# PARAMÉTRAGE DE STREAMLIT
st.title("✅ Fake News Verifier ✅")
user_content = st.text_input(
    "Votre question",
    value="François Mitterand a t-il été avocat ?"
)

if 'run' not in st.session_state:
    st.session_state.run = False

st.button(
    label="Vérifier",
    on_click=on_button_click,
    disabled=st.session_state.run
)


# COMPORTEMENT DE STREAMLIT LORS D'UNE DEMANDE FAITE
# PAR L'UTILISATEUR
if st.session_state.run:
    with st.spinner('Chargement en cours...'):
        # Vérification du contenu de l'entrée
        if not user_content:
            st.warning("Rentrez un mot clé.")

        # NER et recherche Wikipédia
        wiki = [
            search_wikipedia(keyword)
            for keyword in ner(user_content)
        ]

        # Stockage et requête dans la BDD vectorielle
        try:
            vdb = vdb_creator.create_vdb(wiki)
            result = vdb_creator.query_vdb(user_content, vdb)
        except IndexError:
            result = ''

        # Formatage du prompt à envoyer au LLM
        if result != '':
            user_content = PROMPT_TEMPLATE.format(
                context=result,
                question=user_content
            )

        # Envoi du prompt à l'API Groq
        generated_titles = get_groq_completions(user_content)

    # Affichage de la réponse
    st.success("Réponse générée !")
    st.markdown("### Réponse:")
    st.text_area("", value=generated_titles, height=200)

    if st.button("Reposer une question", on_click=on_reset):
        st.stop()
