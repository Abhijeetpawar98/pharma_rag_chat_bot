import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st

# Load environment variables
load_dotenv()

os.environ['GROQ_API_KEY'] = ""  # Replace with your actual Groq API key
os.environ['HF_TOKEN'] = ""      # Replace with your actual Hugging Face API key

# Build FAISS index from PDFs
def build_vector_store(pdf_dir: str) -> FAISS:
    loader = PyPDFDirectoryLoader(pdf_dir)  #PDF Loader to load all the Pdf from the directory
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=250) #Splitting the data
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')  #For embedding
    return FAISS.from_documents(docs, embeddings)  #Storing into FAISS


@st.cache_resource
# Initialize a history-aware RAG chain
def init_conversational_rag_chain(pdf_dir: str):
    store = build_vector_store(pdf_dir)
    retriever = store.as_retriever(search_kwargs={'k': 5, "fetch_k": 16})

    # Reformulation prompt
    reform_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given chat history and a follow-up question, rephrase it as a standalone question understandable without history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        # ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.environ['GROQ_API_KEY']),
        ChatGroq(model="gemma2-9b-it", groq_api_key=os.environ['GROQ_API_KEY']),
        retriever,
        reform_prompt
    )

    # Instruction for LLM Model
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
    You are a highly professional PDF Q&A specialist. The Context block you receive may include plain text, numbered or bulleted lists, and **tables**.  

    1. **Comprehensive coverage**  
       - Treat every piece of Context—text or tables—as equally important.  
       - Always aim to give a **detailed** answer that draws on all relevant Context.  

    2. **Table handling**  
       - If the user’s question touches table data, locate the matching row(s).  
       - For each match, return **all** columns **and** their values, labeled (e.g. “OrderID: 1234; Date: 2025‑04‑20; Amount: ₹2,500; Status: Paid”).  
       - You may also render it as a one‑row mini‑table if that’s clearer.  

    3. **Source fidelity**  
       - Do **not** use any outside knowledge—answer **only** from the provided Context.  

    4. **Graceful fallback**  
       - If the Context does **not** contain the answer, reply in your own professional words, for example:  
         `Sorry, the given context does not contain that information.`  

    5. **Tone & style**  
       - Always be concise yet thorough, as a domain expert would be.  
       - Maintain a friendly but professional chatbot voice.

    Context:
    {context}
    """
        ),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ])

    doc_chain = create_stuff_documents_chain(
        # llm=ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.environ['GROQ_API_KEY']),
        llm=ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.environ['GROQ_API_KEY']),
        prompt=qa_prompt
    )

    # Base RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

    # Session histories
    session_histories: dict[str, BaseChatMessageHistory] = {}
    def get_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in session_histories:
            session_histories[session_id] = ChatMessageHistory()
        return session_histories[session_id]

    # Wrap in RunnableWithMessageHistory
    return RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )