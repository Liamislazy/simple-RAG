# 🤖 Simple Local RAG System

A lightweight, 100% offline Retrieval-Augmented Generation (RAG) pipeline built using Python. This project parses documents, creates semantic vector embeddings, performs similarity lookups, and generates context-aware answers completely locally.

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Vector Math:** NumPy (Cosine Similarity calculated manually)
- **Local LLM & Embeddings:** [Ollama](https://ollama.com/) running `llama3.2` and `nomic-embed-text`
- **Data Source:** *Alice’s Adventures in Wonderland* by Lewis Carroll

---

## 🚀 Key Features & Architectural Decisions

### 1. In-Memory Vector Storage
To optimize retrieval speeds, document embeddings are pre-computed exactly once upon script execution. This avoids calculating $O(N)$ embeddings on every single user query, bringing lookup time down to milliseconds.

### 2. Multi-threaded Concurrency
Embedding 100+ documents sequentially can cause an HTTP network bottleneck. This pipeline uses Python's `concurrent.futures.ThreadPoolExecutor` to hit the local Ollama API endpoints in parallel, speeding up initialization by up to 4x.

### 3. Smart Chunking & Overlap Strategy
Initially, the text sliding-window configuration utilized an overly aggressive stride (`chunk_size=1000`, `stride=100`). This caused severe document inflation (1,634 chunks for a short book) due to a 90% data overlap. The system was optimized to a balanced stride (`stride=800`), reducing the total chunk footprint to ~180 unique segments while preserving semantic continuity.

---

## 📦 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/simple-RAG.git](https://github.com/YOUR_GITHUB_USERNAME/simple-RAG.git)
   cd simple-RAG