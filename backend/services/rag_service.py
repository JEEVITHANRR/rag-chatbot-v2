import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

PERSIST_DIR = "./chroma_db"

_embeddings = None
_vectorstore = None
_qa_chain = None
_uploaded_docs = []


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
    return _embeddings


def process_document(file_path: str, filename: str) -> int:
    global _vectorstore, _qa_chain

    ext = os.path.splitext(filename)[1].lower()
    loader = PyPDFLoader(file_path) if ext == ".pdf" else TextLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    emb = _get_embeddings()

    if _vectorstore is None:
        _vectorstore = Chroma.from_documents(
            chunks, emb, persist_directory=PERSIST_DIR
        )
    else:
        _vectorstore.add_documents(chunks)

    _vectorstore.persist()
    _qa_chain = None  # reset so chain rebuilds on next query

    doc_meta = {
        "name": filename,
        "chunks": len(chunks),
        "pages": len(documents),
    }
    _uploaded_docs.append(doc_meta)
    return len(chunks)


def get_uploaded_docs():
    return _uploaded_docs


def _build_chain():
    global _qa_chain, _vectorstore

    if _vectorstore is None:
        if os.path.exists(PERSIST_DIR):
            _vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=_get_embeddings(),
            )
        else:
            return None

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    _qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
    )
    return _qa_chain


def query_documents(question: str) -> dict:
    global _qa_chain

    if _qa_chain is None:
        _qa_chain = _build_chain()

    if _qa_chain is None:
        return {
            "answer": "No documents have been uploaded yet. Please upload a PDF or text file to get started.",
            "sources": [],
        }

    result = _qa_chain({"question": question})

    sources = []
    for doc in result.get("source_documents", []):
        src = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "")
        label = src + (f" · page {page + 1}" if page != "" else "")
        sources.append(label)

    return {
        "answer": result["answer"],
        "sources": list(dict.fromkeys(sources)),  # deduplicate preserving order
    }
