import streamlit as st
from chatbot import Chatbot
from welcome_page import AccueilPage
from disc import DiscussionPage

def main():
    # Initialiser st.session_state.conversations si nécessaire
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
    if 'current_session' not in st.session_state:
        st.session_state['current_session'] = 1

    st.sidebar.title("Navigation")
    options = ["welcome_page", "disc"]
    choice = st.sidebar.radio("Choisir une page:", options)

    # Créer une instance du chatbot
    chatbot = Chatbot()

    # Créer des instances des pages avec le chatbot
    welcome_page = AccueilPage(chatbot)  
    disc = DiscussionPage(chatbot)

    # Appeler les méthodes display des instances
    if choice == "welcome_page":
        welcome_page.display()  # Afficher la page d'accueil
    elif choice == "disc":
        disc.display()  # Afficher la page de discussion

if __name__ == "__main__":
    main()
