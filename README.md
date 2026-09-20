# 🎓 University Academic Knowledge Assistant

A Retrieval-Augmented Generation (RAG) chat assistant for university
students — ask questions about admissions, fees, exam rules, scholarships,
and academic calendars, and get answers grounded in official documents,
with source + page citations.

**Stack:** Streamlit · FAISS · HuggingFace embeddings · Groq (`openai/gpt-oss-120b`)

---

## Architecture

```
                        ┌────────────────────────────┐
                        │   OFFLINE (Google Colab)    │
                        │  build_faiss_index.py       │
                        │                              │
   Google Drive folder  │  1. Download source docs     │
   (source documents) ─▶│  2. Convert non-PDF → PDF     │
                        │  3. Extract text per page     │
                        │  4. Chunk + embed             │
                        │  5. Save faiss_index/         │
                        └──────────────┬───────────────┘
                                       │  commit index only
                                       │  (raw documents are
                                       │   NEVER pushed to GitHub)
                                       ▼
                        ┌────────────────────────────┐
                        │   GITHUB REPO                │
                        │  app.py, rag_engine.py,      │
                        │  requirements.txt,           │
                        │  faiss_index/ (embeddings)   │
                        └──────────────┬───────────────┘
                                       │  deploy
                                       ▼
                        ┌────────────────────────────┐
                        │  STREAMLIT COMMUNITY CLOUD   │
                        │                              │
   Student question ───▶│  1. Load faiss_index/         │
                        │  2. Embed question, retrieve │
                        │     top-k chunks              │
                        │  3. Build prompt w/ citations │
                        │  4. Call Groq (gpt-oss-120b)  │
   Cited answer     ◀───│  5. Return answer + sources   │
                        └────────────────────────────┘
```

**Why the index is committed but the raw documents aren't:** the FAISS
index (`index.faiss` + `index.pkl`) contains vector embeddings and text
chunks, not the original PDFs — keeping the repo small, avoiding any
document-redistribution concerns, and making deploys fast since nothing
needs to be re-processed at runtime.

---

## Repository structure

```
university-rag-assistant/
├── app.py                       # Streamlit chat UI
├── rag_engine.py                 # Retrieval + Groq generation logic
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example      # template — real secrets.toml is gitignored
└── faiss_index/                  # embeddings only, built in Colab
    ├── index.faiss
    └── index.pkl
```

---

## Setup

### 1. Build the embeddings (once, in Google Colab)
Run `build_faiss_index.py` in Colab. It downloads your source documents
from Google Drive, chunks and embeds them, and produces a `faiss_index/`
folder. Download it and place it at the repo root as shown above.

### 2. Local development
```bash
git clone <your-repo-url>
cd university-rag-assistant
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and add your real GROQ_API_KEY

streamlit run app.py
```

### 3. Get a Groq API key
Sign up at [console.groq.com](https://console.groq.com) and create an API
key. Groq's free tier is sufficient for development and light use.

---

## Deploy to Streamlit Community Cloud

1. Push this repo (including `faiss_index/`) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and
   select this repo with `app.py` as the entry point.
3. Under **App settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-real-key-here"
   ```
4. Deploy. First load will be slower while the embedding model downloads
   and caches — subsequent loads are fast.

---

## Notes & known constraints

- **Memory:** Streamlit Community Cloud's free tier has ~1GB RAM.
  `sentence-transformers` + `BAAI/bge-small-en-v1.5` fit comfortably, but if
  you scale to a much larger document set or a bigger embedding model,
  consider swapping local embeddings for the HuggingFace Inference API to
  cut memory usage.
- **Embedding model consistency:** the model used to build `faiss_index/`
  in Colab (`BAAI/bge-small-en-v1.5`) MUST match the one loaded in
  `rag_engine.py` — mismatched models silently produce poor/irrelevant
  retrieval, not errors.
- **Re-indexing:** whenever source documents change, re-run the Colab
  notebook and replace `faiss_index/` in the repo — the app doesn't
  re-embed anything at runtime.
- **`langchain-community` deprecation:** this package was sunset in mid-2026.
  It's still used here because FAISS has no standalone LangChain package
  yet — this is safe but worth revisiting if/when that changes.
