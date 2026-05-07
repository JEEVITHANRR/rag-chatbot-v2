# RAG AI Chatbot — Streamlit Deployment

A production-grade Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB, and Google Gemini.

## 🚀 Deployment Instructions

### 1. GitHub Setup
- Push this entire project to a new GitHub repository.
- Ensure `requirements.txt` and `streamlit_app.py` are in the root directory.
- The `.gitignore` will prevent your `.env` and local database from being uploaded.

### 2. Streamlit Community Cloud (streamlit.io)
- Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
- Select your repository and set the main file to `streamlit_app.py`.
- **IMPORTANT**: Before clicking Deploy, go to **Advanced Settings** -> **Secrets**.
- Add your Google API Key like this:
  ```toml
  GOOGLE_API_KEY = "your_key_here"
  ```
- Click **Deploy**!

## 🛠️ Local Development
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `GOOGLE_API_KEY` to a `.env` file.
4. Run: `streamlit run streamlit_app.py`

## 📦 Tech Stack
- **Frontend/Backend**: Streamlit
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (Local)
- **Vector Store**: ChromaDB
- **Framework**: LangChain
