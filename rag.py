import os
import signal
import sys
import tempfile
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Embedding uploaded PDFs into ChromaDB...")

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure Gemini API
genai.configure(api_key="<Add your Gemini API key here>")
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Signal handler for graceful termination
def signal_handler(sig, frame):
    print("\nThanks for using the RAG App with Gemini!")
    sys.exit(0)

#signal.signal(signal.SIGINT, signal_handler)

# Function to embed uploaded PDFs into ChromaDB
def embed_uploaded_pdfs(uploaded_files, app_context):
    """
    Embeds the content of uploaded PDF files into ChromaDB.

    Args:
        uploaded_files (list): List of uploaded PDF files.
        app_context (str): Application context for tagging documents.
    """
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # Load and split the PDF content
        loader = PyPDFLoader(tmp_path)
        raw_docs = loader.load()
        docs = text_splitter.split_documents(raw_docs)

        # Add metadata and extend the document list
        for doc in docs:
            doc.metadata["app"] = app_context
        all_docs.extend(docs)

    # Add documents to the vectorstore and persist
    if all_docs:
        vectorstore.add_documents(all_docs)
        st.success(f"✅ Embedded {len(all_docs)} chunks into ChromaDB.")

# Function to retrieve relevant context from ChromaDB
def get_relevant_context(query, app_context):
    """
    Retrieves relevant context from ChromaDB for a given query.

    Args:
        query (str): The user's query.
        app_context (str): Application context for filtering documents.

    Returns:
        str: Relevant context as a concatenated string.
    """
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
    results = vectorstore.similarity_search(query, k=4, filter={"app": app_context})
    
    # Handle case where no results are found
    if not results:
        return "No relevant context found in the database."
    
    return "\n".join([doc.page_content for doc in results])

# Function to generate a RAG prompt
def generate_rag_prompt(query, context):
    """
    Generates a prompt for the RAG model using the query and context.

    Args:
        query (str): The user's query.
        context (str): Relevant context retrieved from ChromaDB.

    Returns:
        str: The generated prompt.
    """
    return f"""Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""

# Function to generate an answer using the Gemini model
def generate_answer(prompt):
    """
    Generates an answer using the Gemini model.

    Args:
        prompt (str): The prompt to be passed to the Gemini model.

    Returns:
        str: The generated answer.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="RAG App with Gemini")
st.title("🔍 RAG-Powered PDF Q&A with Gemini")

# App context selection
app_choice = st.selectbox("Choose App Context", ["app_a", "app_b"])

# File uploader for PDFs
uploaded_files = st.file_uploader("📄 Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Embed uploaded PDFs into ChromaDB
if uploaded_files:
    if st.button("Embed Uploaded PDFs"):
        embed_uploaded_pdfs(uploaded_files, app_choice)

# Input field for user query
query = st.text_input("💬 Ask a question:")

# Generate answer for the query
if st.button("Ask"):
    if query.strip() == "":
        st.warning("Enter a question to proceed.")
    else:
        with st.spinner("Thinking..."):
            context = get_relevant_context(query, app_choice)
            if not context:
                st.warning("No relevant context found.")
            else:
                prompt = generate_rag_prompt(query, context)
                answer = generate_answer(prompt)
                st.success("✅ Answer:")
                st.write(answer)