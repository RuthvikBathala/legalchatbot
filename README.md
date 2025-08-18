# 🧑‍⚖️ LegalChatbot — RAG-powered multi-country legal assistant

LegalChatbot is a Streamlit app and modular pipeline that helps users describe a legal situation and receive a **country-aware** summary with references to the underlying texts. It uses Retrieval-Augmented Generation (RAG), a lightweight **domain classifier**, an **intake loop** to collect missing details, and a **FAISS** vector index per country.

> **Disclaimer:** This tool is for educational and engineering purposes only and is **not** a substitute for professional legal advice.

---

## ✨ Features
- **Multi-country corpus**: place country-specific PDFs or TXT files under `data/<country>/` such as `usa`, `india`, `uk`, `canada`, `australia`, `eu`.
- **RAG pipeline**: HuggingFace embeddings → FAISS vector store → LangChain retrieval.
- **Domain classification (multi-label)**: routes a case to one or more legal domains.
- **Intake & follow-ups**: asks clarifying questions to fill missing fields before reasoning.
- **Reasoner**: composes a final answer grounded in retrieved passages.
- **Modular design**: each stage lives under `src/pipeline/` for easy iteration and testing.

---

## 🗂️ Project structure
legalchatbot/
├─ app.py
├─ .env.example
├─ requirements.txt # choose legacy or modern (see below)
├─ data/ # <country> folders with PDFs or TXT
├─ embeddings/ # FAISS indexes saved per country
├─ prompt_temp/ # prompt templates
└─ src/
└─ pipeline/
├─ encoder.py # build embeddings per country or all
├─ domain_classifier.py # classify into legal domains (multi-label)
├─ intake_parser.py # parse user intake into structured fields
├─ intake_formatter.py # format intake for prompts and reasoning
├─ missing_info_handler.py # detect gaps and ask follow-ups
├─ retriever.py # FAISS retrieval helpers
├─ reasoner.py # final LLM reasoning over retrieved chunks
└─ merge_intake_updates.py # merge user updates back into intake JSON


---

## 🚀 Quick start

### 1) Environment
- Python **3.10+** recommended.
- Create a virtual environment and install **one** of the following sets.

**A) Legacy path** — keep old LangChain import paths  
Install:
~~~bash
pip install -r requirements-legacy.txt
~~~

**B) Modern path (recommended)** — migrate to `langchain-community` and `langchain-huggingface`  
Install:
~~~bash
pip install -r requirements-modern.txt
~~~
> See the **LangChain migration** notes below to update imports.

### 2) Configure secrets
Create a `.env` file in the repo root:
~~~env
OPENAI_API_KEY=sk-...
# Optional, defaults to "embeddings" if not set
EMBEDDING_PATH=embeddings
~~~

### 3) Add data
Put your source files here <PDF and/or .txt files>:
data/
usa/
india/
uk/
canada/
australia/
eu/


### 4) Build embeddings
**Option 1 — Module entrypoint (if available in your copy):**
~~~bash
python -m src.pipeline.encoder            # ingest all countries found under data/
# or a single country
python -m src.pipeline.encoder --country usa
~~~

**Option 2 — Small helper script:**
~~~python
# scripts/build_embeddings.py
from src.pipeline.encoder import ingest_country_laws, ingest_all

# EITHER: build a single country
ingest_country_laws("usa", data_dir="data", out_dir="embeddings")

# OR: build everything found under data/
ingest_all(data_dir="data", out_dir="embeddings")
~~~
Then run:
~~~bash
python scripts/build_embeddings.py
~~~

### 5) Run the app
~~~bash
streamlit run app.py
~~~
Open the local URL shown by Streamlit and try a prompt like:  
> “I was stopped for speeding in New York, what are the penalties and what should I do?”

---

## 🧠 How it works
1. **Domain classification** (`domain_classifier.py`)  
   The LLM returns a list of likely legal domains such as `traffic`, `immigration`, `family_law`.
2. **Intake parsing and follow-ups** (`intake_parser.py`, `missing_info_handler.py`)  
   Intake is normalized to structured JSON. Missing fields trigger brief follow-up questions.
3. **Retrieval** (`encoder.py`, `retriever.py`)  
   Documents are embedded with a Sentence-Transformer and stored in **FAISS**. Retrieval pulls the top-k relevant chunks.
4. **Reasoning** (`reasoner.py`, `intake_formatter.py`)  
   A final response is assembled with references to retrieved passages.

---

## 🔁 LangChain migration notes
If you choose the **modern** requirements, update your imports:
~~~python
# OLD (legacy)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# NEW (modern)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
~~~
Keep `langchain` itself on a current version as in `requirements-modern.txt`.

---

## 🛡️ Security and robustness
- Avoid using `eval()` on LLM outputs. Prefer strict JSON:
  ~~~python
  import json
  result = json.loads(model_output)  # after instructing the model to return JSON
  ~~~
- Be careful with `allow_dangerous_deserialization=True` when calling `FAISS.load_local(...)`. Only load your **own** indexes.
- Redact or avoid storing user PII in logs or prompt traces.

---

## 🧪 Troubleshooting
- **Deprecation warnings with LangChain ≥0.2**: switch to the modern import paths above.
- **FAISS version mismatch**: rebuild indexes after upgrading `faiss-cpu`.
- **CPU vs GPU**: default is CPU. If you installed GPU FAISS by mistake, pin to `faiss-cpu`.
- **Long PDFs**: adjust chunk sizes such as 800–1200 characters with 100–200 overlap. Try top-k between 4 and 8.
- **Streamlit not launching**: ensure you run from the repo root and the virtual environment is activated.

---

## 🗺️ Roadmap
- Add more countries and domain-specific prompt packs.
- Strict JSON schemas for all module inputs and outputs.
- Evaluation harness for retrieval hit rate, groundedness, and answer quality.
- Optional REST API wrapper around the pipeline.

---

## 🤝 Contributing
Pull requests and issues are welcome. Open a discussion if you want to add a new country or domain taxonomy.

---

## 📄 License
Choose a license for this repository such as MIT and add a `LICENSE` file in the repo root.

---

## 🙏 Acknowledgments
- LangChain, FAISS, Sentence-Transformers  
- All referenced legal texts belong to their respective publishers or governments.
