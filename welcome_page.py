import streamlit as st
import json

class AccueilPage:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def display(self):
        st.title("Configuration du Chatbot")
        
        # Formulaire de configuration du chatbot
        st.write("## Modifier les Informations du Chatbot")
        self.chatbot.chatbot_name = st.text_input("Nom du Chatbot", value=self.chatbot.chatbot_name, key="chatbot_name")
        self.chatbot.role = st.text_input("Rôle et Objectif du Chatbot", value=self.chatbot.role, key="chatbot_role")
        self.chatbot.company_name = st.text_input("Nom de la Société", value=self.chatbot.company_name, key="chatbot_company_name")
        self.chatbot.activity_domain = st.text_input("Domaine d'Activité", value=self.chatbot.activity_domain, key="chatbot_activity_domain")
        self.chatbot.instructions = st.text_area("Instructions pour le Chatbot", value=self.chatbot.instructions, key="chatbot_instructions")
        self.chatbot.phone_number = st.text_input("Numéro de Téléphone", value=self.chatbot.phone_number, key="chatbot_phone_number")
        self.chatbot.social_media = st.text_input("Réseaux Sociaux", value=self.chatbot.social_media, key="chatbot_social_media")

        st.write("## Téléversement de Fichiers et URL")
        uploaded_files = st.file_uploader("Téléchargez vos fichiers", type=['pdf', 'csv', 'txt', 'xls', 'json'], accept_multiple_files=True, key="file_uploader")
        url = st.text_input("Entrez une URL à traiter", key="url_input")
        custom_data = st.text_area("Entrez des données personnalisées (ex: 'key1: value1 key2: value2')", key="custom_data")

        if st.button("Soumettre & Traiter"):
            if uploaded_files or url:
                # Appel de la méthode pour traiter les fichiers et l'URL
                vector_store_id = self.chatbot.process_files_and_url(uploaded_files, url, st.session_state.get('current_session', 1))
                st.write(f"Vector store créé avec l'ID: {vector_store_id}")

                # Initialiser ou mettre à jour les informations de conversation
                if 'conversations' not in st.session_state:
                    st.session_state.conversations = {}
                if st.session_state.get('current_session') not in st.session_state.conversations:
                    st.session_state.conversations[st.session_state.get('current_session', 1)] = {'history': [], 'custom_data': json.loads(custom_data or '{}')}
                
                # Passer à la page de discussion
                st.session_state.current_page = "discussion"
                st.experimental_rerun()  # Rafraîchir la page pour passer à la page de discussion

            else:
                st.warning("Veuillez télécharger des fichiers ou entrer une URL pour traiter.")


class Chatbot:
    def __init__(self):
        self.chatbot_name = "Mon Chatbot"
        self.role = "Assistant virtuel"
        self.company_name = "Ma Société"
        self.activity_domain = "Technologie"
        self.instructions = "Aider les utilisateurs"
        self.phone_number = "+0000000000"
        self.social_media = "@mon_chatbot"

    def process_files_and_url(self, files, url, session_id):
        # Logique de traitement des fichiers et URL
        return "id_vector_store"


def main():
    # Créer une instance du chatbot
    chatbot_instance = Chatbot()
    
    # Créer une instance de la page d'accueil avec le chatbot
    page = AccueilPage(chatbot_instance)
    
    # Appeler la méthode display de l'instance
    page.display()


if __name__ == "__main__":
    main()
