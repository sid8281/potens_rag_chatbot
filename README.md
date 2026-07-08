**# 📚 RAG Document Question Answering System

A production-ready Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **ChromaDB**, **Sentence Transformers**, and **Google Gemini**. The system answers questions from uploaded documents, provides citations, detects contradictions across documents, supports multilingual queries, and minimizes hallucinations through retrieval-grounded prompting.

---

## 🚀 Features

- 📄 Multi-document PDF ingestion
- 🔍 Semantic search using Sentence Transformers
- 🎯 Cross-Encoder reranking for improved retrieval quality
- 🤖 Gemini-powered grounded question answering
- 📖 Source citations with page numbers and chunk references
- 🌍 Multilingual query support
- ⚠️ Hallucination prevention (answers only from retrieved context)
- 🤝 Human-in-the-loop confidence flag
- 🔄 Contradiction detection between two documents
- 📊 ChromaDB persistent vector database
- 📝 Structured logging
- ⚡ FastAPI REST APIs with Swagger documentation

---

## 🏗️ Architecture

```
                    User Query
                         │
                         ▼
               Language Detection
                         │
                         ▼
               Query Translation
                         │
                         ▼
             Sentence Transformer
                  Embedding
                         │
                         ▼
                  ChromaDB Search
                         │
                         ▼
          Cross Encoder Re-ranking
                         │
                         ▼
              Prompt Construction
                         │
                         ▼
                  Google Gemini
                         │
                         ▼
      Answer + Citations + Confidence
```

---

# Project Structure

```
rag-assessment/
│
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── prompts/
│   ├── utils/
│   ├── config.py
│   └── main.py
│
├── data/
│   └── documents/
│
├── chroma_db/
│
├── requirements.txt
├── README.md
└── .env
```

---

# Technologies Used

| Category | Technology |
|-----------|------------|
| Backend | FastAPI |
| Vector Database | ChromaDB |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Google Gemini |
| Document Parsing | PyPDF |
| Language Detection | Google Translate / Deep Translator |
| Logging | Python Logging |
| Validation | Pydantic |

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/rag-document-qa.git

cd rag-document-qa
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file.

```env
GEMINI_API_KEY=YOUR_API_KEY

CHROMA_DB_PATH=chroma_db

PRELOAD_DOCS_DIR=data/documents

TOP_K=10

CONFIDENCE_THRESHOLD=0.35
```

---

# Running the Application

```bash
uvicorn app.main:app --reload
```

Swagger UI

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## Health

```
GET /health
```

Returns application status.

---

## Ask Question

```
POST /ask
```

Example

```json
{
    "query":"What is attention mechanism?",
    "top_k":5
}
```

Response

```json
{
  "answer":"...",
  "citations":[...],
  "confidence":0.82,
  "language":"English",
  "hitl_required":false
}
```

---

## Ingest Documents

```
POST /ingest
```

Upload one or more PDF documents for indexing.

---

## Contradiction Detection

```
POST /contradict
```

Example

```json
{
    "doc_id_1":"1706.03762v7.pdf",
    "doc_id_2":"2508.06401v2.pdf",
    "topic":"Attention Mechanism"
}
```

The system retrieves relevant passages from both documents and determines whether they contradict each other.

---

# Retrieval Pipeline

1. User submits question
2. Language detection
3. Translation to English (if required)
4. Generate query embedding
5. Retrieve Top-K chunks from ChromaDB
6. Cross-Encoder reranking
7. Build grounded prompt
8. Generate response using Gemini
9. Return answer with citations

---

# Hallucination Prevention

The system minimizes hallucinations by:

- Retrieval-grounded prompting
- Returning **NO_ANSWER_IN_DOCS** when context is insufficient
- Confidence scoring
- Human review flag for low-confidence answers
- Mandatory citation generation

---

# Cross Encoder Re-ranking

After vector search, retrieved passages are reranked using

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This significantly improves retrieval precision by scoring each `(query, passage)` pair using a transformer-based relevance model.

---

# Supported Documents

- PDF
- TXT
- Markdown

---

# Sample Documents

The repository includes research papers related to

- Transformers
- Retrieval-Augmented Generation (RAG)
- PaperQA
- RAPTOR
- Long Context LLMs
- Attention Mechanisms
- Multimodal Retrieval

---

# Future Improvements

- Hybrid BM25 + Dense Retrieval
- Metadata filtering
- OCR for scanned PDFs
- Streaming LLM responses
- Redis caching
- User authentication
- Evaluation using RAGAS
- Docker deployment
- Kubernetes deployment
- Feedback collection dashboard

---

