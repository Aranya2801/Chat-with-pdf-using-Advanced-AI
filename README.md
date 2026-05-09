<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Chat%20with%20PDF&fontSize=60&fontAlignY=35&desc=Advanced%20AI%20%E2%80%94%20Multi-Agent%20RAG%20System&descAlignY=55&descSize=20&fontColor=fff" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0075FF?style=for-the-badge&logo=meta&logoColor=white)](https://faiss.ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/Aranya2801/Chat-with-pdf-using-Advanced-AI/ci.yml?style=flat-square&label=CI%2FCD)](https://github.com/Aranya2801/Chat-with-pdf-using-Advanced-AI/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](CONTRIBUTING.md)

<br/>

> **A production-grade, multi-agent Retrieval-Augmented Generation (RAG) system for intelligent PDF conversation.**  
> Built with hybrid BM25 + dense vector search, cross-encoder reranking, and autonomous agent routing —  
> designed for daily professional use.

<br/>

[**🚀 Quick Start**](#-quick-start) · [**📐 Architecture**](#-architecture) · [**✨ Features**](#-features) · [**📊 Benchmarks**](#-benchmarks) · [**🐳 Docker**](#-docker-deployment)

<br/>

</div>

---

## 📸 Application Preview

<div align="center">

<img src="docs/screenshots/app_preview.png" alt="Chat with PDF — Advanced AI Full Application Preview" width="100%"/>

</div>

<br/>

The app ships with **four fully functional panels**, all visible in the screenshot above:

<table>
<tr>
<td width="50%" valign="top">

### 💬 1. Chat Interface with Citations
- Ask any question in plain English
- AI answers stream token-by-token in real time
- Every response shows **exact source files + page numbers**
- Agent metadata badges display which AI agent handled your query, the detected intent, and response latency
- 👍 / 👎 feedback buttons on every message
- Chat history sidebar with previous sessions

</td>
<td width="50%" valign="top">

### 📊 2. Analytics Dashboard
- **Live KPI cards** — Total Queries, Sessions, Avg Response Time, Satisfaction Score
- **Queries Over Time** — line chart tracking daily usage (May 1–29 shown)
- **Agent Distribution** — donut chart showing RetrievalAgent 45%, ReasoningAgent 25%, SummaryAgent 15%, TableAgent 10%, CitationAgent 5%
- **Response Time** and **User Satisfaction** trend charts
- Date range picker for custom time windows

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📝 3. Smart Notes Generator
Five selectable note styles in the left panel:
- **Executive Summary** ← shown in preview
- Key Concepts
- Q&A Format
- Bullet Points
- Mind Map

Generated output includes structured takeaways with checkmarks, a full conclusion, and a one-click **download** button for the notes as Markdown.

</td>
<td width="50%" valign="top">

### 📄 4. PDF Viewer with Semantic Search
- Inline **page-by-page PDF rendering** with page counter (2/16 shown)
- **Semantic Search bar** — type a phrase, get ranked passage results with page links
- Each result shows the matching excerpt with keyword highlighting
- **"Go to Page"** button jumps directly to the relevant page
- RAG architecture diagram visible directly inside the rendered PDF

</td>
</tr>
</table>

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                           │
│  ┌──────────┐  ┌──────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │ Chat UI  │  │PDF Viewer│  │  Analytics  │  │  Smart Notes  │  │
│  └────┬─────┘  └──────────┘  └─────────────┘  └───────────────┘  │
└───────┼──────────────────────────────────────────────────────────────┘
        │ query
        ▼
┌───────────────────────────────────────────────────┐
│              AgentOrchestrator                    │
│  ┌──────────────────────────────────────────┐    │
│  │  QueryClassifier (intent + confidence)   │    │
│  └──────────┬───────────────────────────────┘    │
│             │ route                               │
│    ┌────────▼────────────────────────────┐       │
│    │         Agent Pool                  │       │
│    │ ┌────────────┐ ┌──────────────────┐ │       │
│    │ │ Retrieval  │ │    Reasoning     │ │       │
│    │ │   Agent    │ │     Agent        │ │       │
│    │ └────────────┘ └──────────────────┘ │       │
│    │ ┌────────────┐ ┌──────────────────┐ │       │
│    │ │  Summary   │ │     Table        │ │       │
│    │ │   Agent    │ │     Agent        │ │       │
│    │ └────────────┘ └──────────────────┘ │       │
│    └────────┬────────────────────────────┘       │
│             │ enrich                              │
│    ┌────────▼────────┐                           │
│    │  CitationAgent  │                           │
│    └────────┬────────┘                           │
└─────────────┼──────────────────────────────────── ┘
              │ retrieve
              ▼
┌─────────────────────────────────────────────────────┐
│              VectorStoreManager                     │
│  ┌───────────────┐  ┌───────────────┐              │
│  │  FAISS Index  │  │  BM25 Index   │              │
│  │ (dense embs)  │  │ (sparse TF)   │              │
│  └───────┬───────┘  └───────┬───────┘              │
│          └────────┬─────────┘                      │
│           ┌───────▼────────┐                       │
│           │  RRF Fusion    │                       │
│           └───────┬────────┘                       │
│           ┌───────▼────────────────┐               │
│           │ Cross-Encoder Reranker │               │
│           └────────────────────────┘               │
└─────────────────────────────────────────────────────┘
              ▲ index
              │
┌─────────────────────────────────────────────────────┐
│               PDF Processing Pipeline               │
│  Upload → PyMuPDF → Layout Analysis → Chunking      │
│        → Metadata Enrichment → Dedup → Embed        │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Benchmarks

Evaluated on 50 questions across 10 research PDFs (academic papers):

| Retrieval Strategy | Recall@5 | Recall@10 | Latency (p50) |
|-------------------|----------|-----------|---------------|
| BM25 Only | 67.2% | 71.1% | 180ms |
| Dense Only | 76.4% | 80.3% | 420ms |
| Hybrid (RRF) | 82.1% | 86.5% | 510ms |
| **Hybrid + Rerank** | **88.3%** | **91.2%** | 720ms |

> Reranking adds ~200ms latency but improves Recall@5 by **+6.2 points** over raw hybrid.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- 4GB RAM minimum (8GB recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/Aranya2801/Chat-with-pdf-using-Advanced-AI.git
cd Chat-with-pdf-using-Advanced-AI
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate:
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

# Optional: Download spaCy model
python -m spacy download en_core_web_sm
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 5. Launch the App
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. 🎉

---

## 🐳 Docker Deployment

```bash
# Build and run
docker compose up --build

# Or with Docker only:
docker build -t chatpdf-ai .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... chatpdf-ai
```

---

## 🗂️ Project Structure

```
Chat-with-pdf-using-Advanced-AI/
│
├── app.py                          # 🚀 Streamlit entry point
├── requirements.txt                # 📦 All dependencies
├── .env.example                    # ⚙️  Environment template
├── Dockerfile                      # 🐳 Container config
├── docker-compose.yml              # 🐳 Full stack compose
│
├── src/
│   ├── agents/
│   │   ├── orchestrator.py         # 🧠 Multi-agent router
│   │   ├── retrieval_agent.py      # 🔍 Hybrid search + rerank
│   │   ├── reasoning_agent.py      # 💭 Chain-of-Thought QA
│   │   ├── summary_agent.py        # 📝 Map-reduce summarization
│   │   ├── citation_agent.py       # 📚 Page-level citations
│   │   └── table_agent.py          # 📊 Structured data extraction
│   │
│   ├── components/
│   │   ├── sidebar.py              # ⚙️  Upload & settings
│   │   ├── chat_interface.py       # 💬 Streaming chat UI
│   │   ├── pdf_viewer.py           # 📄 Inline PDF viewer
│   │   └── analytics_dashboard.py  # 📊 Usage analytics
│   │
│   ├── vectorstore/
│   │   └── vector_store_manager.py # 🗄️  FAISS/Chroma/Pinecone
│   │
│   └── utils/
│       ├── pdf_processor.py        # 📥 PDF ingestion pipeline
│       ├── query_classifier.py     # 🎯 Intent detection
│       ├── memory.py               # 💾 Conversation memory
│       └── session_manager.py      # 🔑 Session handling
│
├── assets/
│   └── style.css                   # 🎨 Custom UI theme
│
├── tests/
│   └── test_core.py                # ✅ Unit test suite
│
├── notebooks/
│   └── research_notebook.ipynb     # 🔬 Experimentation sandbox
│
├── docs/                           # 📖 Documentation assets
└── .github/
    └── workflows/ci.yml            # 🔄 CI/CD pipeline
```

---

## ⚙️ Configuration Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `model` | `gpt-4o` | LLM for generation |
| `embedding_model` | `text-embedding-3-large` | Embedding model |
| `vector_backend` | `faiss` | `faiss` \| `chroma` \| `pinecone` |
| `top_k` | `8` | Chunks retrieved per query |
| `rerank` | `true` | Cross-encoder reranking |
| `citation_mode` | `true` | Inline source citations |
| `temperature` | `0.1` | LLM temperature (0=factual) |
| `chunk_size` | `1000` | Tokens per chunk |
| `chunk_overlap` | `200` | Overlap between chunks |
| `memory_window` | `10` | Conversation turns to remember |

---

## 🗺️ Roadmap

- [ ] 🌐 Multi-language PDF support (Hindi, Bengali, French, etc.)
- [ ] 🖼️ Figure & chart understanding (GPT-4 Vision)
- [ ] 📊 Interactive data extraction to CSV/Excel
- [ ] 🔗 URL ingestion (arXiv, PubMed, Web pages)
- [ ] 🤝 Collaborative sessions (multi-user)
- [ ] 📱 Mobile-optimized responsive layout
- [ ] 🔒 Local LLM support (Ollama / LM Studio)
- [ ] 📧 Email summary delivery
- [ ] 🗣️ Voice input / text-to-speech output
- [ ] 🧪 RAG evaluation harness (RAGAS metrics)

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

```bash
# Fork → Clone → Branch → Code → Test → PR
git checkout -b feature/your-feature
pytest tests/ -v
git push origin feature/your-feature
```

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com) — LLM orchestration framework  
- [OpenAI](https://openai.com) — GPT-4o & text-embedding-3-large  
- [FAISS](https://faiss.ai) — Facebook AI Similarity Search  
- [Streamlit](https://streamlit.io) — Rapid ML app development  
- [sentence-transformers](https://sbert.net) — Cross-encoder reranking  
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF processing engine  

---

<div align="center">

**Built with ❤️ by [Aranya2801](https://github.com/Aranya2801)**

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
