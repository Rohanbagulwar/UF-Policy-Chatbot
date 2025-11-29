# UF Policy RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built to answer questions about University of Florida (UF) policies using cutting-edge NLP technologies. This system extracts policies from the UF website, processes them intelligently, and uses a large language model to provide accurate, context-aware answers.

## 🎯 Overview

The UF Policy RAG Chatbot leverages a sophisticated retrieval-augmented generation pipeline to provide intelligent responses to policy-related questions. By combining semantic search with generative AI, it delivers accurate, contextual answers based on official UF policies.

### Key Features

- **Semantic Search**: Uses BGE embeddings for intelligent document retrieval
- **Smart Chunking**: LangChain text splitter for optimal context windows
- **Advanced LLM**: OpenAI OSS 120B model from UF Navigator toolkit
- **Persistent Storage**: ChromaDB for efficient vector database management
- **REST API**: FastAPI with WebSocket support
- **Docker Support**: Easy deployment with containerization

---

## 🏗️ System Architecture

### End-to-End RAG Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    UF Policy RAG Chatbot Flow                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DATA COLLECTION & PREPROCESSING                             │
│     ↓                                                            │
│  └─→ UF Website Scraping                                         │
│      └─→ Extract policies (HTML → JSON)                          │
│          └─→ policy_data.json                                    │
│                                                                  │
│  2. TEXT CHUNKING & SPLITTING                                    │
│     ↓                                                            │
│  └─→ LangChain RecursiveCharacterTextSplitter                    │
│      └─→ Chunk size: 1000 | Overlap: 200                         │
│          └─→ Semantic preservation                               │
│                                                                  │
│  3. EMBEDDING GENERATION                                         │
│     ↓                                                            │
│  └─→ BGE Embedding Model (BAAI/bge-base-en-v1.5)                 │
│      └─→ Dimensions: 768                                         │
│          └─→ 384D for small variant                              │
│                                                                  │
│  4. VECTOR STORAGE                                               │
│     ↓                                                            │
│  └─→ ChromaDB Vector Database                                    │
│      └─→ Persistent storage: ./chroma_db                         │
│          └─→ Collection: "policies"                              │
│                                                                  │
│  5. QUERY PROCESSING                                             │
│     ↓                                                            │
│  └─→ User Question Input                                         │
│      └─→ Convert query to embedding (same model)                 │
│          └─→ Semantic similarity search                          │
│              └─→ Retrieve top-k results (k=3-5)                  │
│                                                                  │
│  6. CONTEXT BUILDING                                             │
│     ↓                                                            │
│  └─→ Build contextual information from retrieved documents       │
│      └─→ Format with metadata and relevance scores               │
│          └─→ Prepare for LLM consumption                         │
│                                                                  │
│  7. PROMPT GENERATION                                            │
│     ↓                                                            │
│  └─→ System Prompt (Role definition)                             │
│      ├─→ Retrieved Context                                       │
│      ├─→ User Question                                           │
│      └─→ Instruction for response generation                     │
│                                                                  │
│  8. LLM GENERATION                                               │
│     ↓                                                            │
│  └─→ OpenAI OSS 120B Model (UF Navigator)                        │
│      └─→ Generate contextual answer                              │
│          └─→ Temperature: 0.1 (factual)                          │
│              └─→ Max tokens: 1000                                │
│                                                                  │
│  9. RESPONSE DELIVERY                                            │
│     ↓                                                            │
│  └─→ Return Answer + Metadata                                    │
│      └─→ Include source documents                                │
│          └─→ Confidence scores                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘|
```

---

## 📊 Chunking Strategy

The project uses **LangChain's RecursiveCharacterTextSplitter** for intelligent text chunking:

### Configuration

```python
# Chunk Configuration
chunk_size = 1000          # Characters per chunk
chunk_overlap = 200        # Overlap between chunks
separator_list = [
    "\n\n",               # Paragraph break (priority 1)
    "\n",                 # Line break (priority 2)
    ". ",                 # Sentence break (priority 3)
    " ",                  # Word boundary (priority 4)
    ""                    # Character level fallback
]
```

### Why This Approach?

1. **Recursive Splitting**: Maintains semantic coherence by splitting at natural boundaries
2. **Overlap Strategy**: The 200-character overlap (20% of chunk size) ensures:
   - Context continuity between chunks
   - Important information isn't split across boundaries
   - Better retrieval when queries span chunk boundaries

3. **Separator Priority**: Respects document structure:
   - Keeps paragraphs together when possible
   - Preserves sentence structure
   - Falls back to word/character level if needed

### Benefits

- ✅ Preserves policy context and meaning
- ✅ Optimal chunk size for embedding models
- ✅ Reduces context window waste
- ✅ Improves retrieval accuracy

---

## 🔧 Technologies Used

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Embeddings** | BGE (BAAI) | v1.5 | Generate semantic vectors |
| **Text Splitting** | LangChain | Latest | Intelligent text chunking |
| **Vector DB** | ChromaDB | 1.2.1+ | Store & retrieve embeddings |
| **LLM** | OpenAI OSS 120B | v1 | Generate contextual responses |
| **API Framework** | FastAPI | 0.121.1+ | REST endpoint creation |
| **Server** | Uvicorn | Latest | ASGI server |
| **Language** | Python | 3.10 | Core programming language |

---

## 📓 Notebooks Overview

The project includes several Jupyter notebooks used for different stages of development:

### 1. **policy_scraper.ipynb** 
**Purpose**: Data Collection & Processing
- Scrapes UF policies from the official UF website
- Extracts policy content (text, URLs, metadata)
- Cleans and normalizes text
- Exports to `policy_data.json` format
- **Output**: Raw policy dataset

### 2. **retrival_notebook.ipynb**
**Purpose**: Initial Retrieval Pipeline Testing
- Tests the vector database retrieval mechanism
- Validates ChromaDB connectivity
- Tests query embedding and similarity search
- Iterative development of retrieval logic
- **Output**: Validation reports

### 3. **retrival_updated_pipeline.ipynb**
**Purpose**: Enhanced Retrieval Pipeline
- Improved retrieval with better query processing
- Tests multi-stage retrieval strategies
- Validates context building
- Tests prompt generation
- **Output**: Optimized pipeline code

### 4. **Rag_workflw.ipynb**
**Purpose**: Complete RAG Workflow Integration
- End-to-end pipeline testing
- Integration of all components:
  - Embeddings
  - Vector storage
  - Retrieval
  - LLM generation
- Performance metrics
- **Output**: Validation of complete flow

### 5. **RAG_chatbot_answers.ipynb**
**Purpose**: Response Quality Testing
- Tests chatbot responses to various policy questions
- Validates answer quality and accuracy
- Tests edge cases and corner cases
- Human evaluation of responses
- **Output**: Response quality metrics

### 6. **RAG_accuracy_score_comparison.ipynb**
**Purpose**: Performance Evaluation
- Compares retrieval accuracy across different configurations
- Metrics: Precision, Recall, F1-Score
- Tests different chunk sizes and overlaps
- Compares embedding models
- **Output**: Performance benchmarks and recommendations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip (Python package installer)
- 4GB+ RAM (for embeddings and LLM)
- CUDA-compatible GPU (optional, for faster embeddings)

### Installation Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/Rohanbagulwar/UF-Policy-Chatbot.git
cd RAG_Project
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key packages installed:**
- `langchain` - Text splitting and orchestration
- `sentence-transformers` - BGE embeddings
- `chromadb` - Vector database
- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `openai` - OpenAI client
- `beautifulsoup4` - Web scraping
- `pydantic` - Data validation


---

## 🏃 Running the Application

### Option 1: Local Development (Direct)

#### Step 1: Ensure Dependencies Are Installed
```bash
pip install -r requirements.txt
```

#### Step 2: Start the Server
```bash
uvicorn app:app --port 8000
```

**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

#### Step 3: Access the Application
- **Frontend**: Open browser → `http://localhost:8000`


#### Step 3: Access the Application
- **Frontend**: `http://localhost:8000`
- **API**: `http://localhost:8000/query`



## 📁 Project Structure

```
RAG_Project/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker containerization
│
├── app.py                             # FastAPI application (main entry point)
├── config.py                          # Configuration classes
├── main.py                            # RetrievalPipeline orchestrator
│
├── models/
│   └── embedding_model.py             # BGE embedding wrapper
│
├── services/
│   ├── retrieval_service.py           # Core retrieval pipeline
│   ├── context_builder_service.py     # Context preparation
│   └── prompt_builder_service.py      # Prompt generation
│
├── clients/
│   └── openai_client.py               # OpenAI API wrapper
│
├── vectorstore/
│   └── chroma_manager.py              # ChromaDB management
│
├── static/
│   └── index.html                     # Frontend UI
│
├── notebooks/                         # Jupyter notebooks for development
│   ├── policy_scraper.ipynb          # Data collection
│   ├── retrival_notebook.ipynb        # Initial retrieval tests
│   ├── retrival_updated_pipeline.ipynb # Enhanced pipeline
│   ├── Rag_workflw.ipynb             # Full workflow
│   ├── RAG_chatbot_answers.ipynb      # Answer quality testing
│   └── RAG_accuracy_score_comparison.ipynb # Performance metrics
│
├── chroma_db/                         # ChromaDB vector storage (auto-created)
│   └── chroma.sqlite3
│
├── bge-small/                         # BGE embedding model (auto-downloaded)
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
│
├── policy_data.json                   # Raw extracted policies
└── policies_data.xlsx                 # Excel export of policies
```

---

## 🔄 RAG Pipeline Components

### 1. **Embedding Model** (`models/embedding_model.py`)
- **Model**: BAAI/bge-base-en-v1.5
- **Dimension**: 768
- **Purpose**: Convert text to semantic vectors
- **Performance**: Fast inference, high quality embeddings

### 2. **Vector Store** (`vectorstore/chroma_manager.py`)
- **Database**: ChromaDB
- **Storage**: SQLite + binary index
- **Operations**: Store, retrieve, search
- **Persistence**: Auto-saves to disk

### 3. **Retrieval Service** (`services/retrieval_service.py`)
- **Process**:
  1. Encode query using same embedding model
  2. Search ChromaDB for top-k similar documents
  3. Apply optional filters
  4. Return results with metadata

### 4. **Context Builder** (`services/context_builder_service.py`)
- **Purpose**: Format retrieved documents for LLM
- **Operations**:
  - Combine multiple documents
  - Add relevance scores
  - Format source citations
  - Prepare metadata

### 5. **Prompt Builder** (`services/prompt_builder_service.py`)
- **Components**:
  - System role definition
  - Retrieved context injection
  - User question formatting
  - Generation instructions

### 6. **LLM Client** (`clients/openai_client.py`)
- **Model**: OpenAI OSS 120B (UF Navigator)
- **Temperature**: 0.1 (factual, low creativity)
- **Max Tokens**: 1000
- **Purpose**: Generate contextual responses

---

## ⚙️ Configuration

All configurations are managed in `config.py`:

```python
# Embedding Configuration
EmbeddingConfig:
  model_name = "BAAI/bge-base-en-v1.5"
  dimension = 768 (auto-detected)

# ChromaDB Configuration
ChromaDBConfig:
  persist_directory = "./chroma_db"
  collection_name = "policies"
  description = "UF Policy documents with embeddings"

# OpenAI Configuration
OpenAIConfig:
  api_key = "<Your API Key>"
  base_url = "https://api.ai.it.ufl.edu"
  model = "gpt-oss-20b"
  temperature = 0.1  # Low for factual responses
  max_tokens = 1000

# Query Configuration
QueryConfig:
  n_results = 5       # Top-5 similar documents
  filter_by_type = None  # Optional metadata filtering
```


open any of the notebooks to test components:
- Test embeddings: `RAG_workflow.ipynb`
- Test retrieval: `retrival_notebook.ipynb`
- Test full pipeline: `Rag_workflw.ipynb`
- Compare performance: `RAG_accuracy_score_comparison.ipynb`

---

## 🧾 Evaluation of Different LLMs

This project includes an evaluation of multiple LLMs to measure answer quality on UF policy questions. The evaluation compares human-written (reference) answers against AI-generated answers using standard automatic metrics and manual inspection.

### Evaluation Strategy

- **Sample collection**: We collected a balanced set of **50 sample questions** drawn from UF policy pages and common policy-related queries encountered by users.
- **Reference answers**: For each question, a human expert (project team) wrote a concise, authoritative reference answer based on the official UF policy text.
- **Model responses**: Each LLM (evaluated models) was prompted via the RAG pipeline to generate an answer using the retrieved context from ChromaDB.
- **Metrics**: We computed automated metrics for each (reference, prediction) pair:
  - **BLEU** (n-gram precision)
  - **ROUGE-1 / ROUGE-2 / ROUGE-L** (recall-oriented overlap)
  - **METEOR**
- **Aggregate scoring**: Per-question metric scores were averaged across the 50 samples and then combined (weighted equally or by chosen heuristic) into an aggregate score to rank models.
- **Manual review**: In addition to automatic metrics, team members reviewed a subset of answers to check for factual correctness, hallucinations, and clarity.

### Results Summary

- The evaluation used the above metrics over the 50-sample set. Visualizations of the results (BLEU / ROUGE / METEOR and aggregate ranking) are included in the repository under `presentation_plots/`.
- Summary ranking (aggregate score):
  - **#1 — Llama**: Aggregate Score ≈ **0.2160**
  - **#2 — GPT**: Aggregate Score ≈ **0.2133**
  - **#3 — Gemma**: Aggregate Score ≈ **0.2102**

These values reflect the averaged evaluation metrics across the 50 questions and were used to produce the model comparison plots included in the repo.

### Where to find the visualizations

- See the `presentation_plots/` directory for the charts used in analysis (ROC/score comparison, ROUGE variants, aggregate ranking). Example plots included:
  - `overall_performance_comparison.png` — shows BLEU / ROUGE-L / METEOR / Aggregate_Score across models
  - `rouge_variants_comparison.png` — ROUGE-1 / ROUGE-2 / ROUGE-L per model
  - `model_ranking_aggregate.png` — Horizontal bar chart with aggregate scores and ranks

### Notes & Interpretations

- The differences in aggregate scores are small on this 50-question sample; for production decisions consider a larger evaluation set and human evaluation focusing on factuality and compliance.
- Automated metrics like BLEU/ROUGE favor surface-level overlap and may not fully capture factual correctness or legal nuance — always include manual spot-checks for policy domains.
- If you plan to re-run the evaluation:
  1. keep the same 50-question test set and reference answers to ensure comparability, and
  2. store model outputs (predictions) and per-question metrics as CSV for reproducible ranking.

---

## 📊 Performance Metrics

### Typical Performance (on standard hardware):

| Metric | Value |
|--------|-------|
| Query Embedding Time | ~50ms |
| Vector Search Time | ~20ms |
| Context Building Time | ~10ms |
| LLM Generation Time | ~2-5 seconds |
| **Total Latency** | **~2.5-5.5 seconds** |
| Embedding Accuracy (top-3) | ~87% |
| Answer Relevance Score | ~0.85 |

### Optimization Tips:

1. **Use GPU for Embeddings**:
   ```python
   embedding_model = SentenceTransformer(
       "BAAI/bge-base-en-v1.5",
       device="cuda"  # GPU acceleration
   )
   ```

2. **Batch Processing**:
   - Process multiple queries together
   - Reduce API calls

3. **Caching**:
   - Cache frequent queries
   - Pre-compute common embeddings

---

## 🐛 Troubleshooting

### Issue 1: ChromaDB Connection Error
```
Error: Cannot connect to ChromaDB
```
**Solution**:
```bash
# Remove old database
rm -rf chroma_db/

# Restart application
uvicorn app:app --port 8000
```

### Issue 2: Embedding Model Too Large
```
Error: CUDA out of memory
```
**Solution**:
```python
# Use smaller model in config.py
model_name = "BAAI/bge-small-en-v1.5"  # 33M parameters instead of 110M
```

### Issue 3: OpenAI API Errors
```
Error: Invalid API Key
```
**Solution**:
- Verify API key in `config.py`
- Check UF Navigator toolkit access
- Ensure correct base URL

### Issue 4: Port Already in Use
```
Error: Port 8000 already in use
```
**Solution**:
```bash
# Use different port
uvicorn app:app --port 8001
```

---

## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time policy updates
- [ ] User feedback loop for model improvement
- [ ] Advanced filtering and metadata search
- [ ] Chat history and context management
- [ ] Rate limiting and authentication
- [ ] Performance monitoring dashboard
- [ ] Advanced analytics and reporting

---

## 📝 License

This project is created for EGN-5442 Term project University of Florida.

---

## 👥 Contributors

- **Author**: Rohan Bagulwar 
- **Repository**: [UF-Policy-Chatbot](https://github.com/Rohanbagulwar/UF-Policy-Chatbot)

---


For issues, questions, or suggestions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include error messages and steps to reproduce

---

## 🎓 Learning Resources

### RAG Pipeline Concepts
- [LangChain Documentation](https://docs.langchain.com)
- [ChromaDB Guide](https://docs.trychroma.com)
- [Sentence Transformers](https://www.sbert.net)

### Deployment
- [FastAPI Tutorial](https://fastapi.tiangolo.com)
- [Docker Guide](https://docs.docker.com)
- [Uvicorn Documentation](https://www.uvicorn.org)

### LLMs & Embeddings
- [BGE Embeddings](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [OpenAI API](https://platform.openai.com/docs)

---

**Last Updated**: November 2025

---

## Quick Reference

### Start Server
```bash
pip install -r requirements.txt
uvicorn app:app --port 8000
```

### Docker
```bash
docker build -t uf-policy-chatbot .
docker run -p 8000:8000 uf-policy-chatbot
```

### Test Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the academic integrity policy?"}'
```

---



