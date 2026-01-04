# Compliance Assistant - Intelligent Document Processing System

An agentic AI system that uses Retrieval-Augmented Generation (RAG) to help users query and understand regulatory compliance documents across manufacturing, pharmaceutical, and supply chain industries.

## Project Overview

Compliance teams in regulated industries spend countless hours manually searching through SOPs, FDA guidance documents, OSHA standards, and EPA regulations to answer questions about procedures, safety requirements, and compliance obligations. This project demonstrates how agentic AI can automate this knowledge retrieval process.

### Problem Statement
- Manufacturing, pharmaceutical, and supply chain companies maintain hundreds of compliance documents
- Finding relevant information requires manual search across multiple PDFs and document types
- Knowledge is siloed - related information spans multiple documents
- Time-consuming for employees to get accurate answers to compliance questions

### Solution
An intelligent RAG-based system that:
- Ingests and processes regulatory documents (FDA, OSHA, EPA) and internal SOPs
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
- [ ] Multi-document ingestion (PDF, DOCX)
- [ ] Intelligent text chunking with overlap
- [ ] Local vector similarity search
- [ ] Source citation and metadata tracking
- [ ] Natural language querying
- [ ] Zero external API costs (when using local LLM)

### Planned Enhancements
- [ ] Multi-agent system (retrieval agent + reasoning agent + compliance checker)
- [ ] Conversational memory for follow-up questions
- [ ] Advanced filtering by document type, date, or category
- [ ] Evaluation metrics (RAGAS framework)
- [ ] Export responses to PDF reports
- [ ] User authentication and query logging

## Tech Stack

### Core Technologies
- **Python 3.9+**
- **LangChain** - Agent orchestration and RAG pipeline
- **ChromaDB** - Vector database (local, no external dependencies)
- **Sentence Transformers** - Free, local embedding generation
- **Streamlit** - Web interface

### LLM Options (All Free!)
- **Ollama** - Run Llama 3.1/3.3 locally (recommended)
- **Google Gemini API** - 1500 free requests/day

### Document Processing
- **PyPDF2** - PDF text extraction
- **python-docx** - DOCX processing
- **RecursiveCharacterTextSplitter** - Intelligent text chunking

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

**First query to try:** *"What are the cleaning validation requirements for pharmaceutical manufacturing equipment?"*

---

## Project Structure
```
compliance-assistant/
│
├── data/
│   ├── raw/                       # Original documents
│   │   ├── fda_guidance/         # FDA regulatory guidance
│   │   ├── osha_standards/       # OSHA safety standards
│   │   └── company_sops/         # Sample company SOPs
│   └── processed/                 # Processed/chunked data
│
├── vectorstore/                   # ChromaDB persistence
│
├── src/
│   ├── document_processor.py     # PDF/DOCX ingestion & chunking
│   ├── embeddings.py             # Embedding generation
│   ├── vectorstore_manager.py    # ChromaDB operations
│   ├── retriever.py              # Retrieval logic
│   └── agent.py                  # Agent orchestration
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

## Data Sources

This project uses publicly available regulatory documents:

### Included Sample Documents
- **FDA Guidance**: Sterile products, process validation, equipment cleaning
- **OSHA Standards**: Machine guarding, lockout/tagout, hazard communication
- **EPA Guidelines**: Medical waste, pharmaceutical waste disposal
- **Sample SOPs**: Mock company standard operating procedures

### Adding Your Own Documents
1. Place PDFs or DOCX files in `data/raw/` (or subdirectories)
2. Run document processor: `python src/document_processor.py`
3. Rebuild vector store: `python src/vectorstore_manager.py`

**Supported formats:** PDF, DOCX

## Testing

Run unit tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_document_processor.py -v
```

## Performance Considerations

### Current Performance
- **Document Processing**: ~50-100 pages/second
- **Embedding Generation**: ~1000 chunks/minute (local CPU)
- **Query Latency**: <2 seconds for retrieval + inference

### Optimization Tips
- Use GPU for faster embedding generation
- Adjust `CHUNK_SIZE` based on document structure
- Increase `TOP_K_RESULTS` for more comprehensive answers
- Use Groq API for 10x faster inference vs. local

## License

MIT License

---


