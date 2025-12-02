# **AI Knowledge Search Engine**

Full Stack Retrieval Augmented Generation System with Model Comparison, Critique Analysis, and Automated Insights

This project implements a production ready Retrieval Augmented Generation (RAG) system with a full stack architecture. Users can upload documents, run context aware queries, compare multiple large language models, and evaluate responses through a custom critique and similarity-analysis framework. The system is deployed on a Linux VPS using Nginx and systemd, accessible via public IP.

---

## **1. Overview**

This application integrates a FastAPI backend, a React + TypeScript frontend, and multiple Groq hosted LLMs. It provides an end to end system with:

* Custom ingestion and embedding pipeline
* JSONL based vector store (no LangChain or LlamaIndex)
* Dynamic model selection
* Parallel model comparison
* Multi round critique and prompt-quality analysis
* Multi similarity vector scoring
* Automated insight generation
* Context visualization with relevance highlighting
* Stateful history management
* VPS deployment without domain/HTTPS requirements

---

## **2. Functional Features**

### **2.1 Document Upload and Ingestion**

Uploaded documents undergo a multi-stage pipeline:

* Text extraction and normalization
* Chunk segmentation
* Embedding generation using Groq hosted embedding models
* Storage of embeddings and metadata in a JSONL based vector store

### **2.2 Retrieval-Augmented Question Answering**

For each query:

1. The system embeds the question.
2. Performs similarity search over the vector store.
3. Retrieves the top relevant chunks.
4. Constructs a context grounded prompt.
5. Sends the prompt to the selected model.
6. Returns the answer along with exact retrieved context.

### **2.3 Supported Language Models**

Supported Groq hosted models include:

* **LLaMA 3.1 8B** (fast)
* **LLaMA 3.3 70B** (quality)
* **GPT-OSS 20B** (normal)
* **GPT-OSS 120B** (large)
* **LLaMA 4 Maverick 17B** (preview)
* **Qwen3 32B** (multilingual)
* **Groq Compound Model** (system)

Model selection is handled dynamically by the backend.

### **2.4 Model Comparison**

A dedicated comparison mode allows:

* Executing identical queries on two selected models
* Independent retrieval for each model
* Parallel inference
* Side by side answer presentation
* Separate context cards showing which sources influenced each model

### **2.5 Critique Engine**

A secondary LLM evaluates both the model generated answer and the user’s prompt.

#### **Answer Evaluation**

The critique engine assesses:

* Correctness
* Completeness
* Faithfulness to retrieved context
* Hallucination risk
* Clarity and structure

#### **Prompt Evaluation**

The system identifies issues such as:

* Missing context
* Vague or ambiguous phrasing
* Unclear intent or audience
* Lack of format specification
* Multi part questions

It outputs standardized issue tags (Eg. `missing_context`, `too_vague`, `no_format_specified`) and provides a suggested improved prompt.

### **2.6 Automated Insights**

The insights module generates:

* High level summaries
* Key takeaways
* Topic breakdown
* Recommended follow-up questions

### **2.7 Context Transparency and Highlighting**

Retrieved chunks are displayed with:

* Highlighted relevant segments
* Metadata indicating chunk origin
* Expandable context cards

### **2.8 Interaction History**

The application stores past interactions in the session including:

* Queries
* Answers
* Comparisons
* Critiques
* Insights
* Uploaded documents

---

## **2.9 Research Oriented Evaluation Features**

### **2.9.1 Multi Round Critique Loop**

The system implements a critique pipeline in which a secondary LLM evaluates both:

* **the user’s original question**, and
* **the answer produced by the selected base model**.

The critique module identifies issues such as missing context, vague phrasing, unclear intent, and multi part prompts. It generates:

* **structured correctness assessments**
* **faithfulness and hallucination checks**
* **clarity and completeness evaluations**
* **standardized issue tags**
* **improved prompt suggestions**

### **2.9.2 Multi Similarity Vector Scoring**

The system computes multiple similarity metrics for retrieved chunks:

* **Cosine similarity**
* **Dot product similarity**
* **Negative Euclidean (L2) distance**
* **Negative Manhattan (L1) distance**
* **Hybrid weighted scoring (Cosine + Jaccard keyword)**

### **2.9.3 JSON Based Evaluation Dashboard**

Critique outputs and similarity metrics are stored in a lightweight JSON structure containing:

* Prompt issue tags
* Critique summaries
* Per chunk similarity scores
* Model specific evaluation metadata

---

## **3. Backend Architecture (FastAPI)**

Key API endpoints:

* **POST /ingest** — process and index uploaded documents
* **POST /ask** — retrieval + LLM answering
* **POST /compare** — dual-model inference
* **POST /critique** — structured critique pipeline
* **POST /insights** — insight generation
* **GET /documents** — list indexed documents

The backend uses FastAPI with Pydantic for schema validation.
A `systemd` service ensures persistent uptime on the VPS.

---

## **4. Frontend Architecture (React + TypeScript)**

The frontend provides:

* Model selection UI
* Query interface
* Model comparison view
* Critique and insights panels
* Context highlighting components
* Interaction history
* Loading and error states
* SPA routing via Nginx fallback
* Build system via Vite and styling with Tailwind CSS

---

## **5. Deployment and Infrastructure**

The system is deployed using:

* **Ubuntu Server VPS**
* **Nginx** reverse proxy
* Static hosting for the React build
* **systemd** service for backend
* CORS configuration
* SPA handling using `try_files`
* Public IP access

---

## **6. Technology Stack**

**Frontend**

* React
* TypeScript
* Vite
* Tailwind CSS

**Backend**

* FastAPI
* Python
* Uvicorn
* Pydantic
* Groq API

**Infrastructure**

* Linux (Ubuntu)
* Nginx
* systemd
* VPS deployment

---

## **7. Summary**

This project implements a complete RAG based AI knowledge system combining:

* Custom ingestion and vector storage
* Retrieval based answering
* Multiple selectable LLMs
* Parallel model comparison
* Multi round critique and evaluation
* Multi similarity vector scoring
* Automated insight analysis
* Transparent context visualization
* Full stack web architecture
* Live VPS deployment
