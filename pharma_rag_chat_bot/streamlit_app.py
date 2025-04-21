import uuid
import streamlit as st
from rag_backend import *

# Streamlit App Interface
st.set_page_config(page_title="PDF QA Chatbot", layout="wide")
st.title("RAG Chatbot with Memory")

# Unique session ID Every time the User get into App
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize chain
with st.spinner("Loading PDFs"):
    rag_chain = init_conversational_rag_chain(pdf_dir='./pdfs')

# Chat history storage
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

# Handle new queries
if query := st.chat_input("Ask me anything about the PDFs…"):
    # Save user message
    st.session_state.chat_history.append({'role': 'user', 'content': query})
    with st.chat_message('user'):
        st.write(query)

    # Get response
    with st.chat_message('assistant'):
        result = rag_chain.invoke(
            {'input': query},
            config={'configurable': {'session_id': st.session_state.session_id}}
        )
        answer = result['answer']
        st.write(answer)

    # Save assistant response
    st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
