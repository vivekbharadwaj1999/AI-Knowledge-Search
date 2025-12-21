# VivBot: A Document Knowledge Search and RAG Experiment Platform

**Full Stack Retrieval Augmented Generation System with Multi Model Analysis, Faithfulness Evaluation, and Systematic Experimentation**

An RAG research platform featuring 9 embedding models, 10 LLMs, 5 similarity metrics, automated quality assessment, faithfulness tracking, counterfactual analysis, and batch experimentation capabilities. Built for research, evaluation, and systematic experimentation.

---

## What Makes This Different

Unlike typical RAG chatbots, VivBot is a comprehensive research experimentation platform that enables:

- **Multi Model Comparison**: Test 9 embedding models and 10 LLMs simultaneously
- **Explainability First**: See exactly why each chunk was retrieved with 5 different similarity metrics
- **Faithfulness Analysis**: Evidence coverage, hallucination detection, and sentence level support tracking using NLTK tokenization
- **Answer Stability Analysis**: Quantify consistency across retrieval strategies using semantic similarity + ROUGE-L
- **Counterfactual Testing**: Stress test retrieval by modifying chunk selection (remove top, reverse order, random shuffle, etc)
- **Batch Experimentation**: Systematic parameter grid evaluation with automated faithfulness metrics
- **Self Correction Engine**: Automated critique with quality scoring and iterative improvement and prompt coaching
- **Knowledge Extraction**: Generate AI report, knowledge graphs, and cross document analyses
- **User Isolation**: Guest mode for instant access, accounts for persistent research data
- **Research Infrastructure**: Complete experiment logging, reproducibility, and export capabilities

**900+ possible configurations** (9 embeddings × 10 LLMs × 5 metrics × 20 Top K values) with full explainability, faithfulness tracking, and counterfactual analysis.

---

## Key Features at a Glance

### Multi Model Analysis
- **9 Embedding Models**: From 384D (MiniLM) to 3072D (OpenAI), including SBERT, BGE, E5, INSTRUCTOR, GTE, Jina AI
- **10 LLM Options**: Llama 3.1/3.3/4, GPT-OSS, Qwen3, Kimi K2 (up to 256k context)
- **5 Similarity Metrics**: Cosine, Dot Product, L2/L1 Distance, Hybrid (semantic + keyword)

### Operations Dashboard
- **Three Operation Modes**:
  - **Ask**: Single model with multi-metric retrieval comparison
  - **Compare**: Multiple models side-by-side across all metrics
  - **Critique**: Self-correction with quality scoring
 
### Advanced Analysis
- **Answer Stability Metrics**: Semantic (cosine) + ROUGE-L F1 comparison across retrieval methods
- **Query Embedding Visualization**: See the actual vectors (up to 3072D)
- **Explainability**: Every chunk shows scores from all 5 similarity methods

### Faithfulness & Groundedness Evaluation
- **Evidence Coverage**: Percentage of answer sentences supported by retrieved chunks
- **Hallucination Risk**: Percentage of sentences with no supporting evidence
- **Sentence Level Evidence**: Confidence scores per sentence with supporting chunks
- **Citation Coverage**: Percentage of sentences with direct source citations
- **NLTK Tokenization**: NLTK sentence splitting

### Counterfactual Retrieval
- **Remove Top Chunk**: Test dependence on highest ranked result
- **Remove Top 3**: Stronger stress testing by removing top results
- **Reverse Order**: Test if chunk ranking order matters
- **Random Shuffle**: Test stability under randomized chunk order
- **Robustness Metrics**: Answer similarity (semantic/ROUGE-L/Jaccard), chunk overlap, retrieval dependence scores

### Batch Evaluation Grid
- **Systematic Experimentation**: Test multiple questions across parameter combinations
- **Grid Parameters**: Configure similarity methods, Top K values (1-20), vector normalization
- **Operations**: Run Ask, Compare (model pairs), or Critique (answer/critic pairs)
- **Automated Metrics**: Faithfulness analysis for all configurations
- **Export Results**: Download complete evaluation data as JSON

### Quality Assessment Engine
- **Multi Round Critique**: Up to 2 rounds of self correction
- **4D Quality Scoring**: Correctness, Completeness, Clarity, Hallucination Risk (0-5 scale)
- **Prompt Engineering**: Issue tags + improved prompt suggestions
- **Experiment Logging**: JSONL logs with full history and delta metrics

### Knowledge Tools
- **Document Reports**: Executive summaries, concept explanations, study paths, knowledge graphs
- **Cross Document Relation Analysis**: Relationship extraction between multiple documents
- **Insights Generation**: Entities, keywords, follow up questions, mindmaps
- **Practice Q&A**: Auto generated knowledge check questions

### User Management & Data Isolation
- **Guest Mode**: Instant access with ephemeral session storage (auto cleanup on browser close)
- **User Accounts**: Persistent storage for documents, embeddings, and operation logs
- **Data Isolation**: Complete separation between users, guests stored separately
- **Operations Logging**: Automatic logging of all Ask/Compare/Critique operations with JSON export

### Document Processing
- **6 File Formats**: PDF, DOCX, PPTX, XLSX, TXT, CSV
- **Intelligent Chunking**: Configurable size/overlap with boundary awareness
- **Per Document Model Tracking**: Each document remembers its embedding model

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React + TypeScript UI                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │  Upload  │  │  Scope   │  │Operation │  │   Advanced  │  │
│  │  Panel   │  │  Panel   │  │  Panels  │  │   Analysis  │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │
│  ┌─────────────────────────┐  ┌──────────────────────────┐  │
│  │  Batch Evaluation Grid  │  │  Operations Logging      │  │
│  └─────────────────────────┘  └──────────────────────────┘  │
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
│  │  Faithfulness (NLTK) → Evidence Coverage → Metrics   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Counterfactual → Stress Test → Robustness Metrics   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Batch Grid → Parameter Sweep → Export Results       │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Auth → Guest/User Isolation → Operations Logging    │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Analyze → Critique → Report → Relations → Insights  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Tech Stack:**
- **Frontend**: React, TypeScript, Vite, Tailwind CSS, Headless UI
- **Backend**: FastAPI, Python 3.10+, Uvicorn
- **ML/AI**: Sentence Transformers, Groq API, OpenAI API (optional), ROUGE, NLTK
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

**First Time Setup:**
1. Open http://localhost:5173 (starts in guest mode automatically)
2. Upload a document (PDF, DOCX, TXT, etc.)
3. Choose an embedding model (Eg. "bge-base-en-v1.5" for best balance)
4. Try asking a question about your document!

---

## Complete User Guide

### 1. Guest Mode & Account Management

VivBot starts in guest mode for instant access. Create an account to save your uploaded documents and logs permanently.

**Guest Mode (Default)**
- When you open VivBot, you automatically start as a guest with full feature access
- **Guest data is temporary**: All uploaded documents, embeddings, and operation logs are deleted when you close the browser

**Create an Account**
- Click 'Login/Signup' in the top right to create an account
- **Account benefits**: Your documents, vector embeddings, and operation logs persist forever
- You can logout and login anytime without losing data

**Data Isolation**
- Each user's data is completely separate
- Guests and logged in users each have their own private storage

**Account Management**
- **Logout (logged-in users)**: Click your username dropdown → 'Logout' to logout from your account and return to guest mode
- **Delete Account**: Logged in users can permanently delete their account and all associated data from the dropdown menu

---

### 2. Upload & Index a Document

Start by uploading a document in section 1: "Upload & index a document". Supported formats include PDF, TXT, CSV, DOCX, XLSX, and PPTX.

**Steps:**
1. Click "Choose File" and pick a document from your computer
2. Choose an **Embedding model** (local free options or OpenAI paid embeddings)
   - This choice affects how your document is represented in vector space
   - Changing it requires re-indexing (Upload & Index again)
3. Adjust **Chunk size** to control how documents are split before indexing
   - Smaller chunks give finer grained retrieval
   - Larger chunks preserve more context
4. Set **Chunk overlap** to repeat part of the previous chunk to preserve context across boundaries
   - Changing chunking settings also requires reuploading the document
5. Press "Upload & Index" to chunk the file and create vector embeddings
6. Once indexed, the document appears in the dropdown used by the later sections

---

### 3. Documents & Search Scope

Use section 2 to decide which documents are used when answering questions or generating reports.

**Options:**
- Choose "All documents" to search across all uploaded documents, or select a single document from the dropdown
- **"Generate AI Report"**: Creates a structured explanation of the selected document
- **"Relations between these documents"**: Explores how multiple documents are related in the vector space
- **"Batch Analysis"**: Allows you to choose multiple metrics, and ask multiple questions to perform experiments in a batch.
- **"Remove all documents"**: Removes all uploaded documents so you can start over

---

### 4. Similarity Functions, Top K and Vector Normalization

For experiments, VivBot lets you choose how similarity between embeddings is measured and how many Top K relevant chunks are retrieved.

**Similarity Metrics**

You can pick the similarity function used to rank context chunks in all operations (Relations, Ask, Compare, Critique, Batch Evaluation).

Available metrics include:

1. **Cosine similarity**: Measures the angle between two vectors. High when the embeddings point in the same direction, regardless of magnitude.
   - Formula: `(x · y) / (‖x‖ ‖y‖)`

2. **Negative Manhattan distance (L1)**: Uses the sum of absolute differences between coordinates. High (good) when vectors differ only slightly across many dimensions. Resilient to noise.
   - Formula: `−Σ |xᵢ − yᵢ|`

3. **Negative Euclidean distance (L2)**: Uses the straight line distance between vectors. Punishes large individual deviations strongly, making mismatches stand out.
   - Formula: `−√(Σ (xᵢ − yᵢ)²)`

4. **Dot product**: Measures magnitude × alignment. Favors vectors that are both similar and high energy (large norms).
   - Formula: `Σ xᵢ yᵢ`

5. **Hybrid (Cosine + Jaccard)**: Blends semantic similarity (cosine) with token overlap similarity (Jaccard). Rewards embeddings that match in meaning and share lexical structure.
   - Formula: `α·cosine(x,y) + (1−α)·(|A ∩ B| / |A ∪ B|)`

**Impact**
- Changing the metric can affect which chunks are selected, how similarity is grounded, and how highlight rankings behave

**Top K Configuration**
- Set "Top K" to control how many relevant chunks are retrieved from the vector store
- **Top K** sets how many top ranked document chunks are selected
- Higher values give the model more context, lower values make retrieval stricter
- Range: 1-20 chunks

**Vector Normalization**
- Use **Vector normalization** to control whether embeddings are L2 normalized before scoring
- When enabled, cosine and dot product behave equivalently
- When disabled, dot product also reflects vector magnitude

---

### 5. Ask Questions (Grounded Q&A)

Section 3 answers questions using only the selected document(s) or all documents as context.

**Steps:**
1. Check the "Answering for document …" text to see which document is active
2. Pick an LLM (for example LLaMA 3.1 8B Instant)
3. Type a question about the selected document and press "Ask"
4. The answer appears on the right, together with the retrieved context

**Highlight Context**

Use the **Highlight Context** after using Auto Insights on each answer card to inspect context chunks and check highlighted parts of it:

- **AI**: highlights the exact spans the model seems to rely on most
- **Keywords**: highlights chunks that best match your query terms
- **Sentences**: highlights the most similar sentences in each chunk
- **Off**: hides all highlighting if you just want to read the context

---

### 6. Auto Insights

Auto Insights provides a higher level layer of reasoning on top of Q&A and your uploaded documents. It transforms raw answers and source chunks into structured insights.

**Features:**
- Creates a short **summary** that captures the key ideas from the retrieved context
- Extracts important **entities** such as people, organisations, technologies, frameworks, and locations
- Identifies relevant **keywords** to give a quick sense of the document's focus areas
- Generates **'Suggested Questions'**, that help you explore deeper or prepare for interviews/presentations
- Builds a compact **'Mindmap style'** text representation that connects different topics of the content

---

### 7. Compare Models

Section 4 lets you compare how two different LLMs answer the same question.

**Steps:**
1. Enter a question in the "Compare models" box
2. Choose Model A and Model B from the dropdowns
3. VivBot uses the same retrieved context for both models, so the comparison is fair
4. The Output panel shows a side by side card with both answers and their sources

---

### 8. Critique Answer & Prompt

Section 5 analyses both the prompt and the model answer, and can run a double critique loop.

**Process:**
1. Ask a question you want to analyse more deeply in the "Critique answer & prompt" section
2. VivBot checks for **prompt problems** such as: missing context, vagueness, multi questions, or unclear audience
3. The **answer itself is critiqued** for grounding, reasoning quality, structure, and possible hallucinations

**Double Critique Loop**
- You can enable a **double critique loop**, where one model first critiques the answer and the prompt
- Suggests a new improved prompt
- This is again given back to the answering model to refine the answer
- This pipeline helps you iteratively improve both the prompt and the answer quality

**Output**
- The output shows the differences in the accuracy of the answer from both rounds
- Shows differences in the correctness, hallucinations, and similarity of the answer from both rounds

---

### 9. Batch Evaluation

Batch Evaluation runs systematic experiments across multiple questions and configurations to generate comparative data.

**Access**
- Click 'Batch Evaluation' button in Section 2 to open the panel

**Configuration**

**Questions**
- Add multiple questions to test systematically
- Each question will be evaluated across all selected configurations

**Operations**
- Select which operations to run:
  - **Ask**: Single model evaluation
  - **Compare**: Model pairs for side by side comparison
  - **Critique**: Answer/critic model pairs with quality scoring

**Grid Parameters**
- Configure **similarity methods**: Cosine, Dot, L2, L1, Hybrid
- Set **Top K values** (1-20) to test different retrieval depths

**Faithfulness Metrics**
- Enable to calculate evidence coverage, hallucination risk, and sentence level support for each answer

**Vector Normalization**
- Enable to control whether embeddings are L2 normalized before scoring

**Execution**
- **Run evaluation**: Generates a comprehensive results data with all combinations of settings
- Shows answers, metrics, and sources for each configuration
- **Export results**: Download complete evaluation data as JSON for further analysis

---

### 10. Operations Logging & Export

VivBot automatically logs every Ask, Compare, and Critique operation with complete parameter settings for research.

**Automatic Logging**
- Every operation (Ask, Compare, Critique) is logged with all parameters
- Captures: question, models, Top K, similarity metric, document scope, and complete results

**Export Logs**
- Click 'Export logs (JSON)' at the bottom of the operations panel
- Downloads all logged operations as a single JSON file

**Reset Logs**
- Click 'Reset logs' to clear all logged operations and start fresh
- Requires confirmation to prevent accidental deletion

**What Gets Logged**
- Questions and answers
- Context chunks and sources with similarity scores
- Model names and retrieval parameters
- For Critique: all rounds, scores, and improvements

**Compare Logging**
- Compare operations are logged as unified entries with both left and right model results
- Makes analysis easier than separate Ask entries

---

### 11. Advanced Analysis

Advanced analysis runs a deeper, structured breakdown of an existing Ask / Compare / Critique result using the same scope, Top K, and normalization but with every similarity function. It includes Answer Stability analysis to measure how retrieval methods affect outputs.

**How to Use**
- After you run **Ask**, **Compare**, or **Critique**, use the **Advanced analysis** button on that output card
- This generates an additional analysis that helps you inspect the result more deeply

**Answer Stability**
- Shows how consistent answers are across different retrieval methods
- Uses semantic (cosine) and lexical (ROUGE-L) similarity

**Temperature Control**
- Adjust temperature (0-2) and click 'Recompute' to explore how LLM sampling affects answer stability
- Each experiment is saved to history

**Faithfulness & Groundedness**

Analyzes how well answers are supported by retrieved evidence:

- **Evidence Coverage**: Percentage of answer sentences supported by retrieved chunks (higher = better grounding)
- **Hallucination Risk**: Percentage of answer sentences with no supporting evidence (lower = better)
- **Sentence level Evidence**: Shows each answer sentence with its confidence score and supporting chunks, helping identify which parts are well grounded vs potentially hallucinated
- **Citation Coverage**: Percentage of sentences with direct source citations

**Counterfactual Retrieval**

Tests how answers change when retrieval is modified:

- **Remove Top Chunk**: Removes the highest ranked chunk to test dependence on top results
- **Remove Top 3**: Removes the top 3 chunks for stronger stress testing
- **Reverse Order**: Reverses chunk ranking to see if order matters
- **Random Shuffle**: Randomizes chunk order to test stability
- **Metrics**: Answer similarity (semantic, ROUGE-L, Jaccard), chunk overlap, and retrieval dependence score

**Export JSON**
- Includes complete history of all advanced analysis information for research analysis

**Note**
- If your retrieval settings change (scope / Top K / normalization), rerun the operation to analyze the new grounding context

---

### 12. Output Panel on the Right

All results show up as separate cards on the right side of the screen.

**Features:**
- Ask, Compare, AI Reports, Auto Insights, and Critique each create their own card
- You can scroll down through previous results at any time
- Use the **OUTPUT** and **OPERATIONS** buttons at the bottom on mobile to switch between viewing the output panel and the operations panel

---

## Research Applications

### Experimental Capabilities

**Research Questions This Platform Can Address:**

1. **Embedding Model Comparison**: How do different embedding models (384D to 3072D) affect retrieval quality?
2. **Similarity Metric Analysis**: Which similarity metric works best for different document types?
3. **Answer Stability**: How consistent are LLM answers across retrieval strategies?
4. **Faithfulness Evaluation**: How well are answers grounded in retrieved evidence?
5. **Retrieval Robustness**: How do answers change when top chunks are removed or shuffled?
6. **Self Correction Effectiveness**: Does automated critique improve answer quality?
7. **Cross Document Knowledge**: How do LLMs synthesize relationships between documents?
8. **Top K Sensitivity**: How does the number of retrieved chunks affect answer quality?
9. **Model Comparison**: Which LLMs perform best on domain-specific questions?

### Faithfulness & Robustness Research

**New Research Capabilities:**

1. **Faithfulness Evaluation**: Measure how well answers are grounded in retrieved evidence
   - Evidence Coverage: % of answer sentences supported by chunks
   - Hallucination Risk: % of sentences with no supporting evidence
   - Sentence-level Evidence: Confidence scores per sentence
   - Uses NLTK Punkt tokenizer for accurate sentence segmentation

2. **Retrieval Robustness**: Test answer stability under retrieval variations
   - Counterfactual analysis: Remove top chunks, reverse order, shuffle
   - Retrieval Dependence: How much answers rely on specific chunks
   - Compare models: Which LLM is more robust to retrieval changes?
   - Cross-metric stability: Answer consistency across similarity methods

3. **Batch Experimentation**: Systematic evaluation across parameter grids
   - Multiple questions × models × Top K × similarity metrics
   - Automated faithfulness calculation for all combinations
   - Export complete results for statistical analysis
   - Temperature control for sampling variation studies

### Reproducibility Features

- **Complete Parameter Logging**: Every experiment records models, metrics, settings, timestamps
- **JSONL Experiment Logs**: Structured data for analysis in pandas/R
- **JSON Export**: Results exportable for statistical analysis
- **Deterministic Operations**: Configurable random seeds, fixed algorithms
- **User Isolation**: Separate data per user for multi-researcher environments

### Metrics Collected

**Retrieval Metrics**:
- Top-k precision
- Score distributions across 5 methods
- Chunk overlap between retrieval strategies

**Quality Scores** (Critique):
- Correctness (0-5): Factual accuracy
- Completeness (0-5): How fully answered
- Clarity (0-5): Structure and readability
- Hallucination Risk (0-5): Invented information

**Stability Metrics**:
- Semantic similarity (cosine) between answers
- ROUGE-L F1 between answers
- Temperature sensitivity analysis

**Faithfulness Metrics**:
- Evidence Coverage (%)
- Hallucination Risk (%)
- Citation Coverage (%)
- Per-sentence confidence scores

**Counterfactual Metrics**:
- Answer similarity (semantic/ROUGE-L/Jaccard)
- Chunk overlap
- Retrieval dependence score

**Improvement Deltas** (Critique):
- Round-to-round quality changes
- Prompt improvement effectiveness

---

## Project Structure

```
vivbot/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app + 20+ endpoints
│   │   ├── config.py                # 9 embedding + 10 LLM configs
│   │   ├── vector_store.py          # JSONL storage + 5 similarity metrics
│   │   ├── qa.py                    # Q&A + advanced analysis
│   │   ├── critique.py              # Multi round critique engine
│   │   ├── batch_evaluation.py      # Batch experiment grid
│   │   ├── faithfulness.py          # NLTK sentence tokenization + metrics
│   │   ├── extended_analysis.py     # Counterfactual analysis
│   │   ├── operations_log.py        # Automatic operation logging
│   │   ├── auth.py                  # User authentication system
│   │   ├── insights.py              # Insight generation
│   │   ├── report.py                # Document report generation
│   │   ├── relations.py             # Cross document analysis
│   │   ├── ingest.py                # Multi format file processing
│   │   └── schemas.py               # Pydantic models
│   ├── data/
│   │   ├── users/{username}/        # Per user storage
│   │   │   ├── vector_store.jsonl   # Embeddings + metadata
│   │   │   ├── operations_log.jsonl # Operation history
│   │   │   └── uploads/             # Uploaded documents
│   │   └── guests/{guest_id}/       # Ephemeral guest storage
│   ├── requirements.txt
│   └── INSTALLATION.md
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  # Main application
│   │   ├── components/
│   │   │   ├── AdvancedAnalysisModal.tsx  # Advanced analysis UI
│   │   │   ├── BatchEvaluationPanel.tsx   # Batch evaluation grid
│   │   │   ├── UploadPanel.tsx            # Document upload
│   │   │   ├── AskPanel.tsx               # Q&A interface
│   │   │   ├── OutputPanel.tsx            # Results display
│   │   │   ├── ChatPanel.tsx              # Chat interface
│   │   │   ├── ReportPanel.tsx            # Report generation
│   │   │   ├── AuthModal.tsx              # Login/signup
│   │   │   ├── UserDropdown.tsx           # User menu
│   │   │   ├── RelationsOverlay.tsx       # Document relations
│   │   │   └── InstructionsModal.tsx      # Help documentation
│   │   ├── api.ts                   # API client
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
| bge-base-en-v1.5 | 768 | Local | General purpose (recommended) |
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
- **Hybrid**: When exact keyword matches matter (Eg. names, codes)

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

### 4. Faithfulness Analysis Details

**Sentence Level Evidence Support:**
```json
{
  "total_sentences": 15,
  "evidence_coverage": 0.867,
  "hallucination_risk": 0.133,
  "citation_coverage": 0.733,
  "sentence_support": [
    {
      "sentence": "The candidate has 5 years of Python experience.",
      "has_support": true,
      "confidence": 0.92,
      "supporting_chunks": [2, 5],
      "has_citation": true
    },
    {
      "sentence": "They are an expert in blockchain.",
      "has_support": false,
      "confidence": 0.15,
      "supporting_chunks": [],
      "has_citation": false
    }
  ]
}
```

**NLTK Punkt Tokenizer:**
- Unsupervised sentence boundary detection
- Handles abbreviations, decimals, complex punctuation
- 98%+ accuracy on structured text

### 5. Counterfactual Analysis

**Test Retrieval:**

1. **Remove Top Chunk**: Tests dependence on #1 ranked result
2. **Remove Top 3**: Stronger stress test
3. **Reverse Order**: Tests if ranking order matters
4. **Random Shuffle**: Tests stability under randomization

**Metrics:**
```json
{
  "answer_similarity_semantic": 0.78,
  "answer_similarity_rouge_l": 0.65,
  "answer_similarity_jaccard": 0.52,
  "chunk_overlap": 0.40,
  "retrieval_dependence": 0.55,
  "answer_collapsed": false
}
```

**Retrieval Dependence Score:**
- 0-30%: Low dependence (robust)
- 30-70%: Moderate dependence
- 70-100%: High dependence (fragile)

### 6. Critique Engine Details

**Two Phase Analysis:**

**Phase 1 - Answer Evaluation:**
```json
{
  "scores": {
    "correctness": 4.2,      // 0-5, factual accuracy
    "completeness": 3.8,     // 0-5, how fully answered
    "clarity": 4.5,          // 0-5, structure and readability
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

## License

MIT License - see LICENSE for details.

---

## Project Stats

- **Backend**: ~5,000 lines of Python
- **Frontend**: ~250KB TypeScript/React
- **Configurations**: 900+ possible (9 × 10 × 5 × 20)
- **API Endpoints**: 20+
- **Components**: 10+ major frontend components
- **Supported File Types**: 6 (PDF, DOCX, PPTX, XLSX, TXT, CSV)
- **Embedding Dimensions**: 384D to 3072D
- **Similarity Metrics**: 5 (Cosine, Dot, L2, L1, Hybrid)
- **LLM Models**: 10 (Llama, GPT, Qwen, Kimi)
- **Top K Range**: 1-20 chunks

---

**Please drop a mail to bharadwaj.a.vivek@gmail.com if you face any issues. This is my first RAG project, so i'm learning as I build this as well.**
