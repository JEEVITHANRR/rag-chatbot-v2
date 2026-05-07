"""
RAG AI Chatbot — Streamlit Deployment Version
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# ── LangChain / RAG imports ────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_classic.memory import ConversationBufferMemory

load_dotenv()

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="RAG AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ───────────────────────────────────────────
ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}
# In Streamlit Cloud, we use a temporary directory for the database to ensure it's writable
LOCAL_PERSIST = "./chroma_db_storage"

# ── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0f1117;
    --bg2: #1a1d27;
    --bg3: #22263a;
    --border: rgba(255,255,255,0.08);
    --border2: rgba(255,255,255,0.14);
    --text: #e8eaf0;
    --text2: #8b90a4;
    --text3: #555b72;
    --accent: #4f7fff;
    --accent2: #3a6bff;
    --accent-bg: rgba(79,127,255,0.12);
    --accent-border: rgba(79,127,255,0.3);
    --green: #22c55e;
}

.stApp { font-family: 'Inter', sans-serif !important; }

section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}

.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0 20px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}
.sidebar-brand-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #4f7fff; box-shadow: 0 0 8px rgba(79,127,255,0.5);
}
.sidebar-brand-text { font-size: 16px; font-weight: 600; color: #e8eaf0; letter-spacing: -0.02em; }

.stat-card { background: #1a1d27; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px; }
.stat-label { font-size: 11px; color: #8b90a4; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
.stat-val { font-size: 28px; font-weight: 700; color: #e8eaf0; line-height: 1.1; }

.chat-container { display: flex; flex-direction: column; gap: 16px; padding: 8px 0; }
.msg { display: flex; gap: 12px; max-width: 800px; animation: fadeIn 0.3s ease; }
.msg.user-msg { flex-direction: row-reverse; margin-left: auto; }
.msg-avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; flex-shrink: 0;
}
.msg.user-msg .msg-avatar { background: linear-gradient(135deg, #4f7fff, #3a6bff); color: #fff; }
.msg.bot-msg .msg-avatar { background: #22263a; color: #8b90a4; border: 1px solid rgba(255,255,255,0.14); }
.msg-bubble { padding: 12px 16px; border-radius: 14px; font-size: 14px; line-height: 1.65; }
.msg.user-msg .msg-bubble { background: linear-gradient(135deg, #4f7fff, #3a6bff); color: #fff; border-radius: 14px 4px 14px 14px; }
.msg.bot-msg .msg-bubble { background: #1a1d27; color: #e8eaf0; border: 1px solid rgba(255,255,255,0.08); border-radius: 4px 14px 14px 14px; }

.source-tag {
    font-size: 11px; background: rgba(79,127,255,0.12); color: #4f7fff;
    border: 1px solid rgba(79,127,255,0.3); padding: 2px 10px;
    border-radius: 20px; font-family: 'JetBrains Mono', monospace;
}

.doc-list { background: #1a1d27; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; margin-top: 16px; }
.doc-row { display: flex; align-items: center; gap: 14px; padding: 14px 18px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 13px; }
.chunk-badge { background: rgba(34,197,94,0.1); color: #22c55e; font-size: 11px; padding: 3px 10px; border-radius: 20px; }

.page-header { padding: 4px 0 18px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px; }
.page-title { font-size: 20px; font-weight: 700; color: #e8eaf0; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

#MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State ───────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "bot", "text": "Hello! I'm your RAG assistant. Upload a file to start.", "sources": []}]
if "documents" not in st.session_state:
    st.session_state.documents = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "memory" not in st.session_state:
    st.session_state.memory = None

# ── Functions ───────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

def process_document(file_path: str, filename: str) -> int:
    ext = os.path.splitext(filename)[1].lower()
    loader = PyPDFLoader(file_path) if ext == ".pdf" else TextLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = Chroma.from_documents(chunks, get_embeddings(), persist_directory=LOCAL_PERSIST)
    else:
        st.session_state.vectorstore.add_documents(chunks)
    
    st.session_state.qa_chain = None
    st.session_state.documents.append({"name": filename, "chunks": len(chunks), "pages": len(documents)})
    st.session_state.total_chunks += len(chunks)
    return len(chunks)

def build_chain():
    if st.session_state.vectorstore is None:
        if os.path.exists(LOCAL_PERSIST):
            st.session_state.vectorstore = Chroma(persist_directory=LOCAL_PERSIST, embedding_function=get_embeddings())
        else: return None

    api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY missing. Add it to Streamlit Secrets or .env")
        st.stop()

    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key, temperature=0.2)

    if st.session_state.memory is None:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=st.session_state.memory, return_source_documents=True
    )
    return st.session_state.qa_chain

def query_documents(question: str) -> dict:
    if st.session_state.qa_chain is None: st.session_state.qa_chain = build_chain()
    if st.session_state.qa_chain is None: return {"answer": "Upload a file first!", "sources": []}
    
    result = st.session_state.qa_chain.invoke({"question": question})
    sources = []
    for doc in result.get("source_documents", []):
        src = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "")
        sources.append(f"{src}" + (f" · p{page+1}" if page != "" else ""))
    return {"answer": result["answer"], "sources": list(dict.fromkeys(sources))}

# ── UI ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand"><div class="sidebar-brand-dot"></div><div class="sidebar-brand-text">RAG Chatbot</div></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["💬 Chat", "📄 Documents", "📊 Dashboard"], label_visibility="collapsed")
    if len(st.session_state.documents) > 0:
        st.markdown(f'<div style="margin-top:12px;padding:6px 12px;background:rgba(79,127,255,0.12);border:1px solid rgba(79,127,255,0.3);border-radius:20px;font-size:12px;color:#4f7fff;text-align:center;">📚 {len(st.session_state.documents)} documents loaded</div>', unsafe_allow_html=True)

if page == "💬 Chat":
    st.markdown(f'<div class="page-header"><div class="page-title">💬 Chat</div></div>', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role_class = "user-msg" if msg["role"] == "user" else "bot-msg"
        avatar = "You" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="msg {role_class}"><div class="msg-avatar">{avatar}</div><div><div class="msg-bubble">{msg["text"]}</div>' + (''.join([f'<span class="source-tag">{s}</span>' for s in msg.get("sources", [])])) + '</div></div>', unsafe_allow_html=True)

    if question := st.chat_input("Ask anything…"):
        st.session_state.messages.append({"role": "user", "text": question, "sources": []})
        with st.spinner("Thinking..."):
            res = query_documents(question)
        st.session_state.messages.append({"role": "bot", "text": res["answer"], "sources": res["sources"]})
        st.rerun()

elif page == "📄 Documents":
    st.markdown('<div class="page-header"><div class="page-title">📄 Documents</div></div>', unsafe_allow_html=True)
    if uploaded_file := st.file_uploader("Upload PDF/TXT/MD", type=["pdf", "txt", "md"]):
        if f"proc_{uploaded_file.name}" not in st.session_state:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                chunks = process_document(tmp.name, uploaded_file.name)
            st.session_state[f"proc_{uploaded_file.name}"] = True
            st.success(f"Indexed {uploaded_file.name} ({chunks} chunks)")

    for doc in st.session_state.documents:
        st.markdown(f'<div class="doc-row">📄 <b>{doc["name"]}</b> ({doc["pages"]} pages) <span class="chunk-badge">{doc["chunks"]} chunks</span></div>', unsafe_allow_html=True)

elif page == "📊 Dashboard":
    st.markdown('<div class="page-header"><div class="page-title">📊 Dashboard</div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="stat-card"><div class="stat-label">Docs</div><div class="stat-val">{len(st.session_state.documents)}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card"><div class="stat-label">Questions</div><div class="stat-val">{len([m for m in st.session_state.messages if m["role"]=="user"])}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-card"><div class="stat-label">Chunks</div><div class="stat-val">{st.session_state.total_chunks}</div></div>', unsafe_allow_html=True)
