# Compliance Assistant - Intelligent Document Processing System

A RAG web application that helps users query and understand pharmaceutical regulatory compliance documents.  

## Project Overview

Compliance teams in regulated industries spend countless hours manually searching through FDA guidance documents to answer questions about procedures, safety requirements, and compliance obligations. This project demonstrates how RAG allows users to query large, unstructured document repositories with natural language in less time. 

### Problem Statement
- Finding relevant and accurate information requires manual search across multiple PDFs and document types, which is time-consuming for employees

### Solution
An intelligent RAG-based system that:
- Ingests and processes regulatory documents (e.g, FDA)
- Uses vector similarity search to find relevant information
- Synthesizes answers from multiple documents
- Provides source citations for transparency and auditability

## Architecture
```
User Query
    ↓
Query Processing & Embedding
    ↓
Vector Similarity Search (ChromaDB)
    ↓
Context Retrieval (Top-K Documents)
    ↓
LLM Agent (Reasoning & Synthesis)
    ↓
Response + Citations
```

**Key Components:**
- **Document Processor**: Ingests PDFs/DOCX, chunks text intelligently
- **Embedding Engine**: Converts text to vectors using sentence-transformers
- **Vector Store**: ChromaDB for fast similarity search
- **Retrieval System**: Finds most relevant document chunks
- **Agent Orchestration**: LangChain-based agent for reasoning and synthesis
- **User Interface**: Streamlit web app for demos

## Key Features

### Current Capabilities
- [x] PDF document ingestion
- [x] Intelligent text chunking with overlap
- [x] Local vector similarity search
- [ ] Source citation and metadata tracking
- [x] Natural language querying
- [x] Zero external API costs (when using local LLM)
- [x] Containerize with Docker

### Planned Enhancements
- [ ] Display PDF sources on Streamlit 
- [ ] Conversational memory for follow-up questions
- [ ] Advanced filtering by document type, date, or category
- [ ] Evaluation metrics (RAGAS framework)
- [ ] Export responses to PDF reports
- [ ] User authentication and query logging

## Tech Stack

### Frontend
- **Streamlit** - Web interface

### Backend
- **LangChain** - RAG pipeline
- **ChromaDB** - Local Vector database
- **Sentence Transformers** - Local embedding generation
- **RecursiveCharacterTextSplitter** - Intelligent text chunking
- **Ollama** - Run Llama 3.1/3.3 locally 
- **PyPDF2** - PDF text extraction
- **python-docx** - DOCX processing

## Data Source
- **FDA Guidance**: https://www.fda.gov/regulatory-information/search-fda-guidance-documents

## Installation

### Setup Steps
```bash
# 1. Setup
git clone <repo-url> && cd compliance-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Get data
python download_samples.py

# 3. Process documents
python src/document_processor.py

# 4. Build vector store
python src/vectorstore_manager.py

# 5. Run app
streamlit run app.py
```

### Docker Deployment
```bash
# 1. Clone the repository:
git clone https://github.com/YOUR_USERNAME/compliance-assistant.git
cd compliance-assistant

# 2. Start Ollama on your host machine:
ollama serve 

# 3. Run with Docker Compose:
docker-compose up
```

---

## Project Structure
```
compliance-assistant/
│
├── data/
│   ├── raw/                       # Original documents
│   │   ├── fda_guidance/         # FDA regulatory guidance
│
├── vectorstore/                   # ChromaDB persistence
│
├── src/
│   ├── document_processor.py     # PDF/DOCX ingestion & chunking
│   ├── embeddings.py             # Embedding generation
│   ├── vectorstore_manager.py    # ChromaDB operations
│   ├── retriever.py              # Retrieval logic
│   └── llm_agent.py              # Llama3.1 via Ollama
│
├── notebooks/                     # Jupyter notebooks for exploration
│
├── tests/                         # Unit tests
│
├── app.py                         # Streamlit application
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
├── download_samples.py            # Sample document downloader
└── README.md
```

## Usage

### 1. Process Documents
Ingest and chunk your compliance documents:
```bash
python src/document_processor.py
```

This will:
- Load all PDFs and DOCX files from `data/raw/`
- Split them into semantically meaningful chunks
- Prepare them for embedding

### 2. Build Vector Store
Generate embeddings and populate ChromaDB:
```bash
python src/vectorstore_manager.py
```

### 3. Run the Application
Launch the Streamlit interface:
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### 4. Query Examples

Try these sample queries:
- *"What are the cleaning validation requirements for pharmaceutical equipment?"*
- *"What PPE is required when handling hazardous waste?"*
- *"Explain the lockout/tagout procedure for machinery maintenance"*
- *"What are the storage requirements for sterile drug products?"*

## Configuration

Edit `config.py` to customize:
```python
# Chunking parameters
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval settings
TOP_K_RESULTS = 5          # Number of chunks to retrieve
```

