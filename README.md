# 🗪 OmniScribe AI 

> A Retrieval-Augmented Generation (RAG) application that transforms any YouTube video into a searchable knowledge base. Ask questions in natural language and get grounded answers with timestamp-linked citations.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)

---

## ✨ Features

- 🎥 Extract transcripts from YouTube videos
- 🔍 Semantic search using FAISS vector embeddings
- 💬 Conversational Q&A over video content
- ⏱️ Clickable timestamp citations
- ⚡ Fast inference with Groq LLaMA 3.1
- 🆓 Fully powered by free APIs and local embeddings

---

## 📸 Application Preview

### Home Interface

<p align="center">
  <img width="1906" height="994" alt="image" src="https://github.com/user-attachments/assets/e88d7abf-1e80-4933-8d3b-f593a093c931" />
</p>

### Question Answering

<p align="center">
  <img width="1907" height="998" alt="image" src="https://github.com/user-attachments/assets/879f3819-09f4-4482-a66d-572a92224d62" />
</p>

<p align="center">
<img width="1180" height="841" alt="image" src="https://github.com/user-attachments/assets/a5ffb1f7-e9a9-4f50-9482-dd8a30379ee2" />
</p>

---

## 🏗️ Architecture

```text
YouTube Video
      │
      ▼
Transcript Extraction
      │
      ▼
Chunking + Metadata
      │
      ▼
Embeddings (MiniLM)
      │
      ▼
FAISS Vector Store
      │
      ▼
Semantic Retrieval
      │
      ▼
Prompt Augmentation
      │
      ▼
Groq LLaMA 3.1
      │
      ▼
Answer + Timestamp Citations
```

---

## Tech Stack

| Component             | Technology                |
| --------------------- | ------------------------- |
| Frontend              | Streamlit                 |
| LLM                   | Groq LLaMA-3.1-8B-Instant |
| Framework             | LangChain                 |
| Embeddings            | all-MiniLM-L6-v2          |
| Vector Database       | FAISS                     |
| Validation            | Pydantic                  |
| Transcript Extraction | youtube-transcript-api    |

---

## Project Structure
```
OmniScribe-AI/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .env
├── faiss_cache/
└── assets/
```
---

## Setup & Installation

### Prerequisites
- Python 3.10 or 3.11 recommended (3.13 has known threading quirks with some libraries)
- A free Groq API key from [console.groq.com](https://console.groq.com)
- Git

### Step 1 — Clone the repo
```bash
git clone https://github.com/yourusername/omniscribe-ai.git
cd omniscribe-ai
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
First run downloads the HuggingFace embedding model (~90MB). This happens once.

### Step 4 — Configure your API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=gsk_your_key_here
```

### Step 5 — Run the app
```bash
streamlit run app.py
```
A browser tab opens at `http://localhost:8501`.

---

## How It Works

1.Extract transcript from a YouTube video.
2.Split transcript into timestamp-aware chunks.
3.Generate embeddings using MiniLM.
4.Store vectors locally in FAISS.
5.Retrieve the most relevant chunks for a user query.
6.Inject retrieved context into a prompt.
7.Generate grounded responses using LLaMA 3.1.
8.Display answer with timestamp-linked citations.

---

## 🛠️ Engineering Challenges

### Transcript Retrieval

Handled breaking API changes introduced in newer versions of youtube-transcript-api.

### Streamlit Concurrency

Resolved transcript-fetching issues by moving blocking operations to worker threads.

### Hallucination Control

Implemented strict context-grounded prompting to ensure answers are generated only from retrieved transcript segments.

### Timestamp Citations

Attached metadata to each transcript chunk, enabling direct navigation to the relevant video segment.

---

## 📈 Future Improvements

- Multi-video knowledge bases
- PDF ingestion support
- Playlist indexing
- Pinecone/Qdrant integration
- RAGAS evaluation metrics
- Structured outputs with Pydantic

---

## 🎓 Key Learnings

-Retrieval-Augmented Generation (RAG)
-Vector databases and semantic search
-Embedding models and chunking strategies
-Prompt engineering for grounded generation
-Streamlit application development
-Production-ready Python project structure

---

## Author 

Ananya Acharya
