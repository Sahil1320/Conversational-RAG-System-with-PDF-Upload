import json
import os
import tempfile
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    hf_token = st.secrets.get("HF_TOKEN", None)

if hf_token:
    os.environ["HF_TOKEN"] = hf_token

st.set_page_config(page_title="Conversational RAG", page_icon="📄", layout="centered")
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and ask questions from their content.")

if "store" not in st.session_state:
    st.session_state.store = {}
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None
if "last_indexed_session" not in st.session_state:
    st.session_state.last_indexed_session = None
if "indexed_doc_count" not in st.session_state:
    st.session_state.indexed_doc_count = 0
if "indexed_chunk_count" not in st.session_state:
    st.session_state.indexed_chunk_count = 0
if "last_indexed_at" not in st.session_state:
    st.session_state.last_indexed_at = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "default_session"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

with st.sidebar:
    st.header("⚙️ Controls")
    model_name = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0,
    )
    retrieval_k = st.slider("Retrieved chunks (k)", min_value=2, max_value=8, value=4)
    show_chunks = st.toggle("Show retrieved chunks", value=True)

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        active_session = st.session_state.store.get(st.session_state.current_session_id)
        if active_session is not None:
            active_session.clear()
        st.rerun()

    if st.button("♻️ Reset Vector DB", use_container_width=True):
        st.session_state.rag_chain = None
        st.session_state.uploaded_signature = None
        st.session_state.last_indexed_session = None
        st.session_state.indexed_doc_count = 0
        st.session_state.indexed_chunk_count = 0
        st.session_state.last_indexed_at = None
        st.success("Vector DB reset.")

    if st.session_state.messages:
        chat_export = {
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "messages": st.session_state.messages,
        }
        st.download_button(
            "⬇️ Download Chat JSON",
            data=json.dumps(chat_export, ensure_ascii=True, indent=2),
            file_name="rag_chat_history.json",
            mime="application/json",
            use_container_width=True,
        )

st.subheader("1) Setup")
api_key = st.text_input("Enter your Groq API key:", type="password")
session_id = st.text_input("Session ID", value="default_session")
st.session_state.current_session_id = session_id

st.subheader("2) Upload PDFs")
uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)
if uploaded_files:
    st.caption("Loaded PDFs: " + ", ".join([f.name for f in uploaded_files]))

with st.expander("Advanced indexing options"):
    chunk_size = st.number_input("Chunk size", min_value=300, max_value=3000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=200, step=20)


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


def build_rag_chain(groq_key: str, files, c_size: int, c_overlap: int, k_value: int):
    llm = ChatGroq(groq_api_key=groq_key, model_name=model_name)

    documents = []
    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_pdf_path = tmp_file.name

        try:
            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()
            documents.extend(docs)
        finally:
            os.remove(temp_pdf_path)

    if not documents:
        return None, 0, 0

    documents = [doc for doc in documents if getattr(doc, "page_content", "").strip()]
    if not documents:
        return None, 0, 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(c_size), chunk_overlap=int(c_overlap))
    splits = text_splitter.split_documents(documents)
    splits = [doc for doc in splits if getattr(doc, "page_content", "").strip()]
    if not splits:
        return None, len(documents), 0

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="chroma_db",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If context does not contain the answer, say exactly: "
        "'I could not find this in the uploaded PDF(s).' "
        "Do not use outside knowledge. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ), len(documents), len(splits)


if uploaded_files and api_key:
    uploaded_signature = tuple((f.name, len(f.getvalue())) for f in uploaded_files)
    if (
        st.session_state.uploaded_signature != uploaded_signature
        or st.session_state.last_indexed_session != session_id
    ):
        with st.spinner("Processing PDFs and creating retriever..."):
            result = build_rag_chain(api_key, uploaded_files, chunk_size, chunk_overlap, retrieval_k)
            if result is not None:
                st.session_state.rag_chain, doc_count, chunk_count = result
                st.session_state.indexed_doc_count = doc_count
                st.session_state.indexed_chunk_count = chunk_count
                st.session_state.last_indexed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                st.session_state.rag_chain = None
                st.session_state.indexed_doc_count = 0
                st.session_state.indexed_chunk_count = 0
            st.session_state.uploaded_signature = uploaded_signature
            st.session_state.last_indexed_session = session_id

        if st.session_state.rag_chain:
            st.success("PDFs indexed. You can now ask questions.")
        else:
            st.error("Could not index the uploaded PDFs.")

if not api_key:
    st.warning("Please enter the Groq API Key")
elif not uploaded_files:
    st.info("Upload one or more PDF files to start Q&A.")

if st.session_state.rag_chain:
    col1, col2, col3 = st.columns(3)
    col1.metric("Files", len(uploaded_files) if uploaded_files else 0)
    col2.metric("Chunks", st.session_state.indexed_chunk_count)
    col3.metric("Last indexed", st.session_state.last_indexed_at or "-")

st.subheader("3) Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask from uploaded PDF(s)...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not api_key:
        answer = "Enter Groq API key first."
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    elif not uploaded_files or not st.session_state.rag_chain:
        answer = "Upload PDFs and click 'Create / Refresh Vector DB' first."
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        start = time.process_time()
        response = st.session_state.rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        answer = response.get("answer", "")
        context_docs = response.get("context", [])
        elapsed = round(time.process_time() - start, 2)

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"Response time: {elapsed} sec")

        st.session_state.messages.append({"role": "assistant", "content": answer})

        if not context_docs:
            st.info("No relevant context found in uploaded PDF(s) for this question.")
        elif show_chunks:
            with st.expander("📂 Retrieved Chunks"):
                for i, doc in enumerate(context_docs, start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.caption(
                        f"Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', '?')}"
                    )
                    st.write(doc.page_content)
                    st.write("------")
