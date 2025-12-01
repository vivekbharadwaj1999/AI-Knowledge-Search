# AI Knowledge Search Engine

Full-Stack Retrieval-Augmented Generation System with Model Comparison, Critique Analysis, and Automated Insights

This project implements a production-ready Retrieval-Augmented Generation (RAG) system with a full-stack architecture. Users can upload documents, run context-aware queries, compare multiple large language models, and evaluate responses through a custom critique framework. The system is deployed on a Linux VPS using Nginx and systemd, accessible via public IP without requiring a domain.

## 1. Overview

This application integrates a FastAPI backend, a React + TypeScript frontend, and multiple Groq-hosted LLMs. It provides an end-to-end system with:

* Custom ingestion and embedding pipeline
* JSONL-based vector store (no LangChain or LlamaIndex)
* Dynamic model selection
* Parallel model comparison
* Automated critique and prompt-quality analysis
* Insight generation
* Context visualization with relevance highlighting
* Stateful history management
* VPS deployment without domain/HTTPS requirements

The project demonstrates practical skills in AI engineering, information retrieval, backend and frontend development, and real production deployment.

## 2. Functional Features

### 2.1 Document Upload and Ingestion

Uploaded documents undergo a multi-stage pipeline:

* Text extraction and normalization
* Chunk segmentation
* Embedding generation using Groq-hosted embedding models
* Storage of embeddings and metadata in a JSONL-based vector store

This enables efficient vector similarity search and a lightweight RAG pipeline.

### 2.2 Retrieval-Augmented Question Answering

For each query:

1. The system embeds the question.
2. Performs similarity search over the vector store.
3. Retrieves the top relevant chunks.
4. Constructs a context-grounded prompt.
5. Sends the prompt to the selected model.
6. Returns the answer along with the exact retrieved context.

This ensures transparent, source-grounded responses.

### 2.3 Supported Language Models

The system supports the following Groq-hosted models:

* LLaMA 3.1 8B (fast)
* LLaMA 3.3 70B (quality)
* GPT-OSS 20B (OpenAI OSS)
* GPT-OSS 120B (OpenAI OSS, large)
* LLaMA 4 Maverick 17B (preview)
* Qwen3 32B (multilingual)
* Groq Compound Model (system)

Model selection is handled dynamically by the backend.

### 2.4 Model Comparison

A dedicated model comparison mode allows:

* Executing identical queries on two selected models
* Independent retrieval for each model
* Parallel inference
* Side-by-side answer presentation
* Separate context cards showing which sources influenced each model

This feature supports benchmarking and qualitative evaluation.

### 2.5 Critique Engine

A secondary LLM evaluates both the model-generated answer and the user’s prompt.

#### Answer Evaluation

The critique engine assesses:

* Correctness
* Completeness
* Faithfulness to retrieved context
* Hallucination risk
* Clarity and structure

#### Prompt Evaluation

The system identifies issues such as:

* Missing contextual requirements
* Vague or ambiguous phrasing
* Lack of format specification
* Multi-part or compound questions
* Unclear intent or audience

It outputs standardized issue tags (e.g., `missing_context`, `too_vague`, `no_format_specified`) and provides an improved version of the prompt.

### 2.6 Automated Insights

An insights module generates:

* High-level summaries
* Key takeaways
* Topic breakdown
* Recommended follow-up questions

This produces analysis similar to modern AI-powered research tools.

### 2.7 Context Transparency and Highlighting

Retrieved chunks are displayed with:

* Highlighted relevant segments
* Metadata indicating chunk origin
* Expandable context cards

This makes the RAG process interpretable and verifiable.

### 2.8 Interaction History

The left sidebar stores past interactions including:

* Queries
* Answers
* Comparisons
* Critiques
* Insights
* Uploaded documents

Users can reopen previous interactions seamlessly.

## 3. Backend Architecture (FastAPI)

Key API endpoints:

* POST /ingest — process and index uploaded documents
* POST /ask — retrieval + LLM answering
* POST /compare — dual-model answer generation
* POST /critique — structured evaluation pipeline
* POST /insights — insight generation
* GET /documents — list available documents

The backend uses Pydantic for strict schema validation.
A systemd service ensures continuous availability on the VPS.

## 4. Frontend Architecture (React + TypeScript)

The frontend provides:

* Model selection UI
* Query input interface
* Model comparison view
* Critique and insights views
* Context highlighting components
* State management for history
* Loading and error-state handling
* SPA routing using an Nginx fallback
* Build system via Vite and styling via Tailwind CSS

## 5. Deployment and Infrastructure

The system is deployed using:

* Ubuntu Server VPS
* Nginx reverse proxy
* Static hosting for the React build
* systemd service for backend process management
* CORS configuration
* SPA fallback using `try_files`
* Direct IP-based access with no domain or HTTPS

This setup ensures stability without depending on third-party hosting platforms.

## 6. Technology Stack

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
* VPS deployment (IP-based)

## 7. Summary

This project implements a complete RAG-based AI knowledge system combining:

* Custom ingestion and vector storage
* Retrieval-based answering
* Multiple selectable LLMs
* Parallel model comparison
* Automated critique and insight analysis
* Transparent context visualization
* Full-stack web architecture
* Deployment on a live VPS environment

The system goes beyond typical notebook-based demos and demonstrates practical, production-oriented skills in AI engineering, retrieval system design, full-stack development, and deployment.

---

If you'd like, I can also prepare:

* a shorter minimal README
* a GitHub-optimized README with badges
* or a version tailored specifically for thesis internship applications.
