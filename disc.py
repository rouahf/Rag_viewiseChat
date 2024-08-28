import streamlit as st

class DiscussionPage:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def display(self):
        st.title("Historique des Conversations")
        
        # Vérification de la présence de l'ID du vecteur store
        if 'vector_store_id' not in st.session_state:
            st.error("Veuillez d'abord traiter les fichiers ou l'URL sur la page d'accueil.")
            return
        
        st.write("Posez une question au chatbot")
        user_question = st.text_input("Votre question :")
        session_id = st.session_state.get('current_session', 1)

        if user_question:
            try:
                answer = self.chatbot.handle_user_input(user_question, session_id)
                st.write(f"Réponse : {answer}")
            except ValueError as e:
                st.error(str(e))
