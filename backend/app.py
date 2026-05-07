"""
RAG AI Chatbot — Streamlit Application
Combines backend RAG pipeline and frontend UI into a single Streamlit app.
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
LOCAL_PERSIST = "./chroma_db_local"

# ── Custom CSS (dark theme) ─────────────────────────────
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
    --green-bg: rgba(34,197,94,0.1);
    --amber: #f59e0b;
    --red: #ef4444;
    --red-bg: rgba(239,68,68,0.1);
}

.stApp { font-family: 'Inter', sans-serif !important; }

section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stRadio > label {
    color: var(--text2) !important;
    font-size: 13px !important;
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
.sidebar-footer { font-size: 11px; color: #555b72; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.08); margin-top: 24px; }

.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 20px; }
.stat-card { background: #1a1d27; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px; transition: border-color 0.2s; }
.stat-card:hover { border-color: rgba(79,127,255,0.3); }
.stat-label { font-size: 11px; color: #8b90a4; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
.stat-val { font-size: 28px; font-weight: 700; color: #e8eaf0; line-height: 1.1; }
.stat-sub { font-size: 11px; color: #555b72; margin-top: 4px; }

.chat-container { display: flex; flex-direction: column; gap: 16px; padding: 8px 0; }
.msg { display: flex; gap: 12px; max-width: 800px; animation: fadeIn 0.3s ease; }
.msg.user-msg { flex-direction: row-reverse; margin-left: auto; }
.msg-avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; flex-shrink: 0; margin-top: 2px;
}
.msg.user-msg .msg-avatar { background: linear-gradient(135deg, #4f7fff, #3a6bff); color: #fff; }
.msg.bot-msg .msg-avatar { background: #22263a; color: #8b90a4; border: 1px solid rgba(255,255,255,0.14); }
.msg-bubble { padding: 12px 16px; border-radius: 14px; font-size: 14px; line-height: 1.65; }
.msg.user-msg .msg-bubble { background: linear-gradient(135deg, #4f7fff, #3a6bff); color: #fff; border-radius: 14px 4px 14px 14px; }
.msg.bot-msg .msg-bubble { background: #1a1d27; color: #e8eaf0; border: 1px solid rgba(255,255,255,0.08); border-radius: 4px 14px 14px 14px; }
.msg-sources { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 5px; }
.source-tag {
    font-size: 11px; background: rgba(79,127,255,0.12); color: #4f7fff;
    border: 1px solid rgba(79,127,255,0.3); padding: 2px 10px;
    border-radius: 20px; font-family: 'JetBrains Mono', monospace;
}

.doc-list { background: #1a1d27; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; margin-top: 16px; }
.doc-list-header { padding: 12px 18px; font-size: 12px; font-weight: 500; color: #8b90a4; border-bottom: 1px solid rgba(255,255,255,0.08); text-transform: uppercase; letter-spacing: 0.05em; }
.doc-row { display: flex; align-items: center; gap: 14px; padding: 14px 18px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 13px; transition: background 0.15s; }
.doc-row:last-child { border-bottom: none; }
.doc-row:hover { background: rgba(255,255,255,0.02); }
.doc-icon { width: 34px; height: 34px; background: rgba(79,127,255,0.12); border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; color: #4f7fff; font-size: 14px; }
.doc-name { color: #e8eaf0; font-weight: 500; flex: 1; }
.doc-meta { color: #555b72; font-size: 12px; }
.chunk-badge { background: rgba(34,197,94,0.1); color: #22c55e; font-size: 11px; padding: 3px 10px; border-radius: 20px; font-weight: 500; }

.upload-zone { border: 2px dashed rgba(255,255,255,0.14); border-radius: 14px; padding: 48px 24px; text-align: center; background: #1a1d27; transition: all 0.25s; margin-bottom: 16px; }
.upload-zone:hover { border-color: #4f7fff; background: rgba(79,127,255,0.06); }
.upload-icon { color: #555b72; font-size: 40px; margin-bottom: 10px; }
.upload-title { font-size: 15px; font-weight: 500; color: #e8eaf0; margin-bottom: 4px; }
.upload-sub { font-size: 12px; color: #555b72; }

.page-header { padding: 4px 0 18px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px; }
.page-title { font-size: 20px; font-weight: 700; color: #e8eaf0; margin-bottom: 2px; }
.page-sub { font-size: 13px; color: #8b90a4; }

.empty-state { text-align: center; padding: 40px 20px; color: #555b72; font-size: 13px; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }
.typing-dots span { display: inline-block; width: 7px; height: 7px; background: #555b72; border-radius: 50%; margin: 0 2px; animation: pulse 1.2s infinite; }
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

.alert-success { background: rgba(34,197,94,0.1); color: #22c55e; border: 1px solid rgba(34,197,94,0.25); padding: 12px 16px; border-radius: 8px; font-size: 13px; margin-bottom: 12px; }
.alert-error { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.25); padding: 12px 16px; border-radius: 8px; font-size: 13px; margin-bottom: 12px; }

#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stChatInput"] textarea { font-family: 'Inter', sans-serif !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "bot",
            "text": "Hello! I'm your RAG-powered document assistant. Upload a PDF or text file, then ask me anything about it. 🚀",
            "sources": [],
        }
    ]
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


# ── RAG Service Functions ───────────────────────────────
def get_embeddings():
    if st.session_state.embeddings is None:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    return st.session_state.embeddings


def process_document(file_path: str, filename: str) -> int:
    """Process a document: load → split → embed → store in ChromaDB."""
    ext = os.path.splitext(filename)[1].lower()
    loader = PyPDFLoader(file_path) if ext == ".pdf" else TextLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    emb = get_embeddings()

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = Chroma.from_documents(
            chunks, emb, persist_directory=LOCAL_PERSIST
        )
    else:
        st.session_state.vectorstore.add_documents(chunks)

    # ✅ fixed: .persist() is deprecated in ChromaDB ≥0.4 — removed

    st.session_state.qa_chain = None  # reset chain so it rebuilds

    st.session_state.documents.append(
        {"name": filename, "chunks": len(chunks), "pages": len(documents)}
    )
    st.session_state.total_chunks += len(chunks)
    return len(chunks)


def build_chain():
    """Build (or rebuild) the ConversationalRetrievalChain."""
    if st.session_state.vectorstore is None:
        if os.path.exists(LOCAL_PERSIST):
            st.session_state.vectorstore = Chroma(
                persist_directory=LOCAL_PERSIST,
                embedding_function=get_embeddings(),
            )
        else:
            return None

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY not found. Please set it in your `.env` file.")
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=api_key,
        temperature=0.2,
    )

    if st.session_state.memory is None:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=st.session_state.memory,
        return_source_documents=True,
    )
    return st.session_state.qa_chain


def query_documents(question: str) -> dict:
    """Query the RAG pipeline."""
    if st.session_state.qa_chain is None:
        st.session_state.qa_chain = build_chain()

    if st.session_state.qa_chain is None:
        return {
            "answer": "No documents have been uploaded yet. Please upload a PDF or text file to get started.",
            "sources": [],
        }

    # ✅ fixed: use .invoke() instead of calling the chain directly (LangChain ≥0.1)
    result = st.session_state.qa_chain.invoke({"question": question})

    sources = []
    for doc in result.get("source_documents", []):
        src = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "")
        label = src + (f" · page {page + 1}" if page != "" else "")
        sources.append(label)

    return {
        "answer": result["answer"],
        "sources": list(dict.fromkeys(sources)),  # deduplicate while preserving order
    }


# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-dot"></div>
        <div class="sidebar-brand-text">RAG Chatbot</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["💬 Chat", "📄 Documents", "📊 Dashboard"],
        label_visibility="collapsed",
    )

    doc_count = len(st.session_state.documents)
    if doc_count > 0:
        st.markdown(f"""
        <div style="margin-top:12px;padding:6px 12px;background:rgba(79,127,255,0.12);
                    border:1px solid rgba(79,127,255,0.3);border-radius:20px;
                    font-size:12px;color:#4f7fff;text-align:center;">
            📚 {doc_count} document{'s' if doc_count > 1 else ''} loaded
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-footer">LangChain · ChromaDB · Google Gemini</div>
    """, unsafe_allow_html=True)


# ── Page: Chat ──────────────────────────────────────────
if page == "💬 Chat":
    subtitle = (
        "Upload a document first, then ask questions about it"
        if not st.session_state.documents
        else f"{len(st.session_state.documents)} document{'s' if len(st.session_state.documents) > 1 else ''} loaded — ask anything"
    )
    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">💬 Document Chat</div>
        <div class="page-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

    # Display messages
    chat_html = '<div class="chat-container">'
    for msg in st.session_state.messages:
        role_class = "user-msg" if msg["role"] == "user" else "bot-msg"
        avatar = "You" if msg["role"] == "user" else "🤖"
        chat_html += f"""
        <div class="msg {role_class}">
            <div class="msg-avatar">{avatar}</div>
            <div>
                <div class="msg-bubble">{msg["text"]}</div>
        """
        if msg.get("sources"):
            chat_html += '<div class="msg-sources">'
            for s in msg["sources"]:
                chat_html += f'<span class="source-tag">{s}</span>'
            chat_html += "</div>"
        chat_html += "</div></div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    question = st.chat_input("Ask anything about your documents…")
    if question:
        st.session_state.messages.append({"role": "user", "text": question, "sources": []})
        with st.spinner("Thinking..."):
            result = query_documents(question)
        st.session_state.messages.append({"role": "bot", "text": result["answer"], "sources": result["sources"]})
        st.rerun()


# ── Page: Documents ─────────────────────────────────────
elif page == "📄 Documents":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📄 Documents</div>
        <div class="page-sub">Upload PDF, TXT, or MD files to index for chat</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">☁️</div>
        <div class="upload-title">Upload a document below</div>
        <div class="upload-sub">PDF, TXT, or MD · up to 200 MB</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"], label_visibility="collapsed")

    if uploaded_file is not None:
        key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
        if key not in st.session_state:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            with st.spinner(f"Processing **{uploaded_file.name}**..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                try:
                    chunks = process_document(tmp_path, uploaded_file.name)
                    os.unlink(tmp_path)
                    st.session_state[key] = True
                    st.markdown(f"""
                    <div class="alert-success">
                        ✅ <strong>{uploaded_file.name}</strong> processed — {chunks} chunks indexed successfully.
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    os.unlink(tmp_path)
                    st.markdown(f'<div class="alert-error">❌ Error: {e}</div>', unsafe_allow_html=True)

    docs = st.session_state.documents
    if docs:
        doc_html = f'<div class="doc-list"><div class="doc-list-header">Indexed documents ({len(docs)})</div>'
        for doc in docs:
            pages_text = f"{doc['pages']} page{'s' if doc['pages'] != 1 else ''}"
            doc_html += f"""
            <div class="doc-row">
                <div class="doc-icon">📄</div>
                <div class="doc-name">{doc['name']}</div>
                <div class="doc-meta">{pages_text}</div>
                <span class="chunk-badge">{doc['chunks']} chunks</span>
            </div>"""
        doc_html += "</div>"
        st.markdown(doc_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="doc-list">
            <div class="doc-list-header">Indexed documents (0)</div>
            <div class="empty-state">No documents uploaded yet. Use the uploader above to get started.</div>
        </div>
        """, unsafe_allow_html=True)


# ── Page: Dashboard ─────────────────────────────────────
elif page == "📊 Dashboard":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📊 Dashboard</div>
        <div class="page-sub">Overview of your document intelligence session</div>
    </div>
    """, unsafe_allow_html=True)

    docs_count = len(st.session_state.documents)
    questions_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    chunks_count = st.session_state.total_chunks

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">Documents Loaded</div>
            <div class="stat-val">{docs_count}</div>
            <div class="stat-sub">PDF / TXT / MD files</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Questions Asked</div>
            <div class="stat-val">{questions_count}</div>
            <div class="stat-sub">This session</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Chunks</div>
            <div class="stat-val">{chunks_count}</div>
            <div class="stat-sub">Vector embeddings stored</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    docs = st.session_state.documents
    if docs:
        doc_html = '<div class="doc-list"><div class="doc-list-header">Loaded documents</div>'
        for doc in docs:
            pages_text = f"{doc['pages']} page{'s' if doc['pages'] != 1 else ''}"
            doc_html += f"""
            <div class="doc-row">
                <div class="doc-icon">📄</div>
                <div class="doc-name">{doc['name']}</div>
                <div class="doc-meta">{pages_text}</div>
                <span class="chunk-badge">{doc['chunks']} chunks</span>
            </div>"""
        doc_html += "</div>"
        st.markdown(doc_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="doc-list">
            <div class="doc-list-header">Loaded documents</div>
            <div class="empty-state">No documents loaded yet. Go to Documents to upload.</div>
        </div>
        """, unsafe_allow_html=True)

    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    if user_msgs:
        st.markdown("""
        <div class="doc-list" style="margin-top:16px;">
            <div class="doc-list-header">Recent questions</div>
        """, unsafe_allow_html=True)
        for msg in user_msgs[-5:]:
            truncated = msg["text"][:80] + ("…" if len(msg["text"]) > 80 else "")
            st.markdown(f"""
            <div class="doc-row">
                <div class="doc-icon">💬</div>
                <div class="doc-name">{truncated}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
