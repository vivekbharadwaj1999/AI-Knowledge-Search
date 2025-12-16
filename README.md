# VivBot: A Document Knowledge Search and an RAG Research Platform

**Full Stack Retrieval Augmented Generation System with Multi Model Analysis, Explainability Dashboard, and Research Experimentation**

An RAG experiment platform featuring 9 embedding models, 10 LLMs, 5 similarity metrics, comprehensive explainability, automated quality assessment, and advanced analysis capabilities. Built for research, and experimentation.

---

## What Makes This Different

Unlike typical RAG chatbots, VivBot is a research experimentation platform that enables:

- **Multi Model Comparison**: Test 9 embedding models and 10 LLMs simultaneously
- **Explainability First**: See exactly why each chunk was retrieved with 5 different similarity metrics
- **Answer Stability Analysis**: Quantify consistency across retrieval strategies using semantic similarity + ROUGE-L
- **Self-Correction Engine**: Automated critique with quality scoring and iterative improvement
- **Knowledge Extraction**: Generate study guides, knowledge graphs, and cross-document analyses
- **Research Infrastructure**: Complete experiment logging, reproducibility, and export capabilities

450+ possible configurations (9 embeddings × 10 LLMs × 5 metrics) with full explainability at every step.

---

## Key Features at a Glance

### Multi-Model Intelligence
- **9 Embedding Models**: From 384D (MiniLM) to 3072D (OpenAI), including SBERT, BGE, E5, INSTRUCTOR, GTE, Jina AI
- **10 LLM Options**: Llama 3.1/3.3/4, GPT-OSS, Qwen3, Kimi K2 (up to 256k context)
- **5 Similarity Metrics**: Cosine, Dot Product, L2/L1 Distance, Hybrid (semantic + keyword)

### Advanced Analysis Dashboard
- **Three Operation Modes**:
  - **Ask**: Single model with multi-metric retrieval comparison
  - **Compare**: Multiple models side-by-side across all metrics
  - **Critique**: Self-correction with quality scoring
- **Answer Stability Metrics**: Semantic (cosine) + ROUGE-L F1 comparison
- **Query Embedding Visualization**: See the actual vectors (up to 3072D)
- **Explainability**: Every chunk shows scores from all 5 similarity methods

### Quality Assessment Engine
- **Multi-Round Critique**: Up to 2 rounds of self-correction
- **4D Quality Scoring**: Correctness, Completeness, Clarity, Hallucination Risk (0-5 scale)
- **Prompt Engineering**: 6 issue tags + improved prompt suggestions
- **Experiment Logging**: JSONL logs with full history and delta metrics

### Knowledge Tools
- **Document Reports**: Executive summaries, concept explanations, study paths, knowledge graphs
- **Cross-Document Analysis**: LLM-powered relationship extraction between multiple documents
- **Insights Generation**: Entities, keywords, follow-up questions, mindmaps
- **Practice Q&A**: Auto-generated knowledge check questions

### Document Processing
- **6 File Formats**: PDF, DOCX, PPTX, XLSX, TXT, CSV
- **Intelligent Chunking**: Configurable size/overlap with boundary awareness
- **Per-Document Model Tracking**: Each document remembers its embedding model

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React + TypeScript UI                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │  Upload  │  │  Scope   │  │Operation │  │   Advanced  │  │
│  │  Panel   │  │  Panel   │  │  Panels  │  │   Analysis  │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────────┐
│                     FastAPI Backend                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Ingest → Chunk → Embed (9 models) → Vector Store    │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Query → Retrieve (5 metrics) → LLM (10 models)      │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Analyze → Critique → Report → Relations → Insights  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Tech Stack:**
- **Frontend**: React, TypeScript, Vite, Tailwind CSS, Headless UI
- **Backend**: FastAPI, Python 3.10+, Uvicorn
- **ML/AI**: Sentence-Transformers, Groq API, OpenAI API (optional), ROUGE
- **Document Processing**: pypdf, python-docx, python-pptx, openpyxl, pandas
- **Deployment**: Nginx, systemd, Ubuntu VPS

---

## Quick Start

### Prerequisites
```bash
# Backend
Python 3.10+
pip install -r requirements.txt

# Frontend
Node.js 18+
npm install
```

### Environment Setup
```bash
# backend/.env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional, only for OpenAI embeddings
```

### Running Locally

**Backend:**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev
# Opens at http://localhost:5173
```

---

## API Endpoints

### Core Operations
- `POST /ingest` - Upload and embed documents (with model selection)
- `POST /ask` - Standard Q&A with model/metric selection
- `GET /documents` - List all indexed documents
- `DELETE /documents` - Clear all documents

### Advanced Analysis
- `POST /analyze` - Multi-method analysis (ask/compare/critique modes)
- `POST /critique` - Run critique engine with scoring
- `GET /critique-log-rows` - Export experiment history

### Knowledge Tools
- `POST /insights` - Generate insights from Q&A
- `POST /report` - Generate document study report
- `POST /document-relations` - Cross-document relationship analysis

### Configuration
- `GET /embedding-models` - List available embedding models

---

## Research Applications

### Experimental Capabilities

**Research Questions This Platform Can Address:**

1. **Embedding Model Comparison**: How do different embedding models (384D to 3072D) affect retrieval quality?
2. **Similarity Metric Analysis**: Which similarity metric works best for different document types?
3. **Answer Stability**: How consistent are LLM answers across retrieval strategies?
4. **Self-Correction Effectiveness**: Does automated critique improve answer quality?
5. **Cross-Document Knowledge**: How do LLMs synthesize relationships between documents?

### Reproducibility Features

- **Complete Parameter Logging**: Every experiment records models, metrics, settings, timestamps
- **JSONL Experiment Logs**: Structured data for analysis in pandas/R
- **JSON Export**: Results exportable for statistical analysis
- **Deterministic Operations**: Configurable random seeds, fixed algorithms

### Metrics Collected

- **Retrieval Metrics**: Top-k precision, score distributions across 5 methods
- **Quality Scores**: Correctness, completeness, clarity, hallucination risk (0-5)
- **Stability Metrics**: Semantic similarity (cosine) + ROUGE-L F1 between answers
- **Improvement Deltas**: Round-to-round quality changes in critique mode

---

## Project Structure

```
vivbot/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + 14 endpoints
│   │   ├── config.py            # 9 embedding + 10 LLM configs
│   │   ├── vector_store.py      # JSONL storage + 5 similarity metrics
│   │   ├── qa.py                # Q&A + advanced analysis
│   │   ├── critique.py          # Multi-round critique engine
│   │   ├── insights.py          # Insight generation
│   │   ├── report.py            # Document report generation
│   │   ├── relations.py         # Cross-document analysis
│   │   ├── ingest.py            # Multi-format file processing
│   │   └── schemas.py           # Pydantic models
│   ├── data/
│   │   ├── raw/                 # Uploaded documents
│   │   ├── vector_store.jsonl   # Embeddings + metadata
│   │   └── critique_log.jsonl   # Experiment logs
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx         
│   │   ├── components/
│   │   │   ├── AdvancedAnalysisModal.tsx 
│   │   │   ├── UploadPanel.tsx
│   │   │   ├── AskPanel.tsx
│   │   │   ├── OutputPanel.tsx
│   │   │   ├── ChatPanel.tsx
│   │   │   ├── ReportPanel.tsx
│   │   │   ├── RelationsOverlay.tsx
│   │   │   └── InstructionsModal.tsx
│   │   ├── api.ts               # API client
│   │   └── workspace.ts
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

---

## Advanced Features Deep Dive

### 1. Multi-Embedding Architecture

**9 Models Spanning 3 Orders of Magnitude:**

| Model | Dimension | Type | Strength |
|-------|-----------|------|----------|
| all-MiniLM-L6-v2 | 384 | Local | Fast baseline |
| bge-base-en-v1.5 | 768 | Local | General purpose |
| e5-base | 768 | Local | Efficient |
| multilingual-e5-base | 768 | Local | 100+ languages |
| instructor-large | 768 | Local | Instruction-aware |
| GTE-large-en-v1.5 | 1024 | Local | SOTA, matches OpenAI |
| jina-v2-base-en | 768 | Local | 8K context window |
| text-embedding-3-small | 1536 | API | OpenAI efficient |
| text-embedding-3-large | 3072 | API | OpenAI premium |

**Note**: Each document tracks which embedding model was used, enabling controlled experiments comparing model impact on retrieval quality.

### 2. Five Similarity Metrics Explained

```python
# 1. Cosine Similarity (default)
score = dot(query, chunk) / (norm(query) * norm(chunk))
# Range: [-1, 1], higher = more similar

# 2. Dot Product
score = dot(query, chunk)
# Range: unbounded, higher = more similar

# 3. Negative L2 Distance (Euclidean)
score = -sqrt(sum((q_i - c_i)^2))
# Range: (-∞, 0], closer to 0 = more similar

# 4. Negative L1 Distance (Manhattan)
score = -sum(|q_i - c_i|)
# Range: (-∞, 0], closer to 0 = more similar

# 5. Hybrid (Semantic + Lexical)
score = 0.7 * cosine(query, chunk) + 0.3 * jaccard(query_words, chunk_words)
# Combines semantic vectors with keyword overlap
```

**Use Cases:**
- **Cosine**: General semantic similarity (most common)
- **Dot Product**: When vector magnitudes matter
- **L2/L1**: When distance metrics are preferred
- **Hybrid**: When exact keyword matches matter (e.g., names, codes)

### 3. Answer Stability Framework

**Problem**: Different retrieval methods may surface different context, leading to answer variation.

**Solution**: Quantify stability using two metrics:

1. **Semantic Similarity**: Embed both answers, compute cosine similarity
2. **ROUGE-L F1**: Measure longest common subsequence overlap

**Output**: Stability matrix showing how each method's answer compares to every other method.

```
Example Stability Matrix (Cosine Selected):
           Cosine   Dot    L2     L1     Hybrid
Semantic:  1.000   0.923  0.887  0.845  0.912
ROUGE-L:   1.000   0.834  0.798  0.756  0.821
```

### 4. Critique Engine Details

**Two-Phase Analysis:**

**Phase 1 - Answer Evaluation:**
```json
{
  "scores": {
    "correctness": 4.2,      // 0-5, how factually accurate
    "completeness": 3.8,     // 0-5, how fully answered
    "clarity": 4.5,          // 0-5, how well structured
    "hallucination_risk": 0.8  // 0-5, invented information
  },
  "answer_critique_markdown": "The answer correctly identifies..."
}
```

**Phase 2 - Prompt Evaluation:**
```json
{
  "prompt_issue_tags": ["missing_context", "no_format_specified"],
  "improved_prompt": "Given the context of...",
  "prompt_feedback_markdown": "The prompt could be improved by..."
}
```

**Self-Correction Loop:**
1. Generate answer from original prompt
2. Critique answer + prompt
3. If quality < 95% or hallucination > 5%, use improved prompt
4. Generate new answer
5. Compare round 1 vs round 2 scores

---

### Development Setup
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install pytest black flake8  # Dev dependencies

# Frontend
cd frontend
npm install
npm run lint
npm run test
```

---

## License

MIT License - see LICENSE for details.

---

## Project Stats

- **Backend**: ~3,500 lines of Python
- **Frontend**: ~197KB TypeScript/React
- **Configurations**: 450+ possible (9 × 10 × 5)
- **API Endpoints**: 14
- **Components**: 7 major frontend components
- **Supported File Types**: 6
- **Embedding Dimensions**: 384D to 3072D
