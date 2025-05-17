# RAG-Powered PDF Q&A with Gemini

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs and ask questions about their content using Google's Gemini AI model. The application uses ChromaDB for vector storage and Streamlit for the user interface.

## Features

- 📄 PDF document upload and processing
- 🔍 Semantic search using ChromaDB vector store
- 🤖 Question answering powered by Google's Gemini AI
- 📊 Interactive Streamlit web interface
- 🔄 Support for multiple application contexts
- 💡 Efficient document chunking and embedding

## Prerequisites

- Python 3.8+
- ChromaDB
- Google Gemini API access
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/manuc021/rag-python.git
cd rag-python
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your Google Gemini API key:
- Replace the API key in `rag.py` with your own key from Google AI Studio
- Or set it as an environment variable (recommended)

## Usage

1. Start the Streamlit application:
```bash
streamlit run rag.py
```

2. Using the application:
   - Select an application context (app_a or app_b)
   - Upload one or more PDF files using the file uploader
   - Click "Embed Uploaded PDFs" to process and store the documents
   - Enter your question in the text input field
   - Click "Ask" to get an AI-generated answer based on the content of your PDFs

## Technical Details

### Components

- **Document Processing**: Uses `PyPDFLoader` and `RecursiveCharacterTextSplitter` for efficient PDF processing
- **Embeddings**: Utilizes `HuggingFaceEmbeddings` with the "sentence-transformers/all-MiniLM-L6-v2" model
- **Vector Store**: ChromaDB for efficient similarity search
- **LLM**: Google's Gemini 2.0 Flash model for generating answers
- **UI**: Streamlit for the web interface

### Key Functions

- `embed_uploaded_pdfs()`: Processes and stores PDF content in ChromaDB
- `get_relevant_context()`: Retrieves relevant document chunks for a query
- `generate_rag_prompt()`: Creates prompts for the Gemini model
- `generate_answer()`: Interfaces with the Gemini API to generate responses

## Project Structure

```
rag-python/
├── rag.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
└── chroma_db_nccn/    # Vector store directory (generated)
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
