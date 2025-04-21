# PDF QA Chatbot

A Streamlit-based RAG (Retrieval-Augmented Generation) chatbot that ingests PDF files, indexes them with FAISS & Hugging Face embeddings, and answers user queries—text or tables—using Groq LLMs.

---

## Features

- **PDF Ingestion:** Load all PDFs from a directory.
- **Chunking & Embeddings:** Split documents into overlapping chunks and embed with `all-MiniLM-L6-v2`.
- **Vector Search:** Use FAISS index and history-aware retrieval for follow-up questions.
- **Table-Aware Q&A:** Detect tables in PDFs and return complete rows with column labels when queried.
- **Streamlit UI:** Interactive chat interface with session memory.

---

## Obtaining API Keys

1. **Groq API Key**
   - Go to the Groq developer portal: https://www.groq.com/login
   - Sign up or log in with your account.
   - Navigate to **API Keys** or **Dashboard**.
   - Copy your `GROQ_API_KEY`.

2. **Hugging Face API Token**
   - Visit Hugging Face: https://huggingface.co/
   - Log in or create an account.
   - Go to **Settings → Access Tokens**.
   - Click **New token**, give it a name, and choose **Read** scope.
   - Copy your `HF_TOKEN`.

## Quick Start Guide

Follow these steps to run the PDF QA Chatbot locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/pharma_rag_chat_bot.git
   cd pharma_rag_chat_bot
   ```

2. **Create and activate a virtual environment**
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your PDF files**
   - Create the `pdfs/` directory (if not present):
     ```bash
     mkdir pdfs
     ```
   - Place `Sample-Pharma.pdf` or any other PDFs into `pdfs/`.

5. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Interact with the chatbot**
   - Open `http://localhost:8501` in your browser.
   - Type questions in the chat box and press Enter.
   - 
7. **Stop the app**
   - Press `Ctrl+C` in the terminal to terminate the Streamlit server.

---

## Project Structure

```
pdf-qa-chatbot/
├── pdfs/                  # Place your PDF files here
├── requirements.txt       # Python dependencies
├── streamlit_app.py       # Streamlit UI + RAG pipeline
└── README.md              # This file
```
