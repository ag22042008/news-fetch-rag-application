import os
import re
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="AI Financial Advisor", page_icon="📊", layout="wide")

PERSIST_DIR = "chroma_db"

SUGGESTED_QUESTIONS = [
    "Summarize this company.",
    "Should I invest in this company?",
    "Analyze revenue growth.",
    "Analyze profitability.",
    "Analyze debt.",
    "Analyze cash flow.",
    "What are the major risks?",
    "Future outlook.",
    "Strengths and weaknesses.",
]

RATING_COLORS = {
    "STRONG BUY": "#2E7D32",
    "BUY": "#5C9A5F",
    "HOLD": "#B08D57",
    "SELL": "#B5654A",
    "STRONG SELL": "#A6452C",
}


# ============================
# API key resolution
# ============================
def get_mistral_api_key():
    key = None
    try:
        key = st.secrets.get("MISTRAL_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.environ.get("MISTRAL_API_KEY")
    if key:
        key = key.strip()
        os.environ["MISTRAL_API_KEY"] = key
    return key


MISTRAL_API_KEY = get_mistral_api_key()

if not MISTRAL_API_KEY:
    st.error(
        "MISTRAL_API_KEY is missing. Locally: add it to a .env file. "
        "On Streamlit Cloud: go to Manage app > Settings > Secrets and add "
        "MISTRAL_API_KEY = your_key, then reboot the app."
    )
    st.stop()

# ============================
# Prompt
# ============================
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert Financial Advisor and Equity Research Analyst.
Answer ONLY using the provided context.
If the answer is not present in the document, reply:
"I could not find this information in the uploaded financial report."
Whenever appropriate, analyze:
• Revenue Growth
• Profitability
• Expenses
• Assets
• Liabilities
• Cash Flow
• Debt
• Business Risks
• Future Outlook
If asked whether someone should invest, provide:
Investment Rating:
- Strong Buy
- Buy
- Hold
- Sell
- Strong Sell
Explain every conclusion using evidence from the document.
Mention page numbers whenever available.
Never use outside knowledge.
""",
        ),
        (
            "human",
            """
Context
{context}
Question
{question}
""",
        ),
    ]
)

# ==================================================================================
# STYLE — "The Ledger Room": a financial-archive theme (ink, brass, ledger paper)
# ==================================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink: #1B2430;
    --ink-soft: #232E3D;
    --paper: #EDE7D9;
    --paper-dim: #E2DBC9;
    --brass: #B08D57;
    --moss: #5C7A6A;
    --chalk: #F2EFE6;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg, var(--ink) 0%, var(--ink-soft) 100%); color: var(--chalk); }
#MainMenu, footer, header { visibility: hidden; }

.fr-header {
    border: 1px solid rgba(176,141,87,0.4);
    border-radius: 2px;
    padding: 28px 32px;
    margin-bottom: 24px;
    background: var(--ink-soft);
    position: relative;
}
.fr-header::before {
    content: ""; position: absolute; inset: 6px;
    border: 1px solid rgba(176,141,87,0.25); pointer-events: none;
}
.fr-eyebrow {
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.18em;
    text-transform: uppercase; font-size: 11px; color: var(--brass); margin-bottom: 6px;
}
.fr-title { font-family: 'Lora', serif; font-weight: 600; font-size: 32px; color: var(--chalk); margin: 0; }
.fr-sub { font-size: 14px; color: rgba(242,239,230,0.55); margin-top: 6px; }

section[data-testid="stSidebar"] { background: var(--ink-soft); border-right: 1px solid rgba(176,141,87,0.25); }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { font-family: 'Lora', serif; color: var(--brass); }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] .stCaption { color: rgba(242,239,230,0.7) !important; }
section[data-testid="stSidebar"] label p {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.05em; color: rgba(242,239,230,0.75) !important;
}
section[data-testid="stSidebar"] button {
    background: transparent !important; border: 1px solid var(--brass) !important; color: var(--brass) !important;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; border-radius: 2px !important;
}
section[data-testid="stSidebar"] button:hover { background: rgba(176,141,87,0.12) !important; }
div[data-testid="stFileUploaderDropzone"] {
    background: rgba(176,141,87,0.06) !important; border: 1px dashed rgba(176,141,87,0.5) !important;
}

div[data-testid="stChatMessage"] { background: transparent; padding: 0; }
div[data-testid="stChatMessageContent"] { border-radius: 4px; padding: 4px 2px; }
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
    background: rgba(176,141,87,0.08); border: 1px solid rgba(176,141,87,0.3);
    border-radius: 4px; padding: 14px 18px; color: var(--chalk);
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
    background: var(--paper); border: 1px solid var(--paper-dim); border-radius: 4px;
    padding: 16px 20px; font-family: 'Lora', serif; color: var(--ink); font-size: 16px; line-height: 1.6;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] p { color: var(--ink) !important; }

.fr-rating {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-weight: 700;
    font-size: 13px; letter-spacing: 0.05em; color: #fff; padding: 5px 14px;
    border-radius: 3px; margin: 6px 0 12px 0;
}
.fr-catalog-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--moss); margin: 10px 0 6px 2px;
}
.fr-slip {
    background: var(--paper); border: 1px solid var(--paper-dim); border-left: 3px solid var(--brass);
    border-radius: 2px; padding: 12px 14px; margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: var(--ink); line-height: 1.55;
}
.fr-slip-num {
    display: inline-block; background: var(--brass); color: var(--ink); font-weight: 700;
    font-size: 11px; padding: 1px 7px; border-radius: 2px; margin-right: 8px;
}
.fr-slip-meta { color: var(--moss); font-size: 11px; margin-top: 6px; opacity: 0.85; }

div[data-testid="stExpander"] { background: transparent; border: 1px solid rgba(176,141,87,0.35); border-radius: 3px; }
div[data-testid="stExpander"] summary { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--brass); }

div[data-testid="stChatInput"] { background: var(--ink-soft); border: 1px solid rgba(176,141,87,0.4); border-radius: 4px; }
div[data-testid="stChatInput"] textarea { color: var(--chalk) !important; }

.fr-rule { border: none; border-top: 1px dashed rgba(176,141,87,0.3); margin: 16px 0; }
.fr-empty {
    border: 1px dashed rgba(176,141,87,0.4); border-radius: 4px; padding: 40px 24px;
    text-align: center; color: rgba(242,239,230,0.6); font-family: 'Lora', serif; font-size: 17px; margin-top: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================
# Cached resources
# ============================
@st.cache_resource
def get_embedding_model():
    return MistralAIEmbeddings(api_key=MISTRAL_API_KEY)


def get_llm(model_name, temperature):
    return ChatMistralAI(model=model_name, temperature=temperature, api_key=MISTRAL_API_KEY)


def index_files(uploaded_files, chunk_size, chunk_overlap):
    """Load, chunk, embed, and store multiple uploaded PDFs into the Chroma vector store."""
    embedding_model = get_embedding_model()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = uploaded_file.name  # keep the real filename, not the temp path
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    vectorstore.add_documents(all_chunks)
    st.session_state.vectorstore_ready = True
    st.session_state.indexed_files = st.session_state.get("indexed_files", set()) | {
        f.name for f in uploaded_files
    }
    return len(all_chunks)


def load_existing_vectorstore():
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        embedding_model = get_embedding_model()
        vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
        try:
            if vs._collection.count() > 0:
                return True
        except Exception:
            pass
    return False


def get_retriever(k, fetch_k, lambda_mult):
    embedding_model = get_embedding_model()
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def build_context(docs):
    """Stitch retrieved chunks into context text, labeled with 1-indexed page numbers."""
    context = ""
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        if page != "Unknown":
            page = page + 1
        context += f"\n\n========== Page {page} ({doc.metadata.get('source', 'document')}) ==========\n"
        context += doc.page_content
    return context


def answer_query(query, retriever, llm):
    docs = retriever.invoke(query)
    if not docs:
        return "I could not find this information in the uploaded financial report.", []
    context = build_context(docs)
    final_prompt = PROMPT.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)
    return response.content, docs


def extract_rating(text):
    match = re.search(r"Investment Rating:\s*\**\s*(Strong Buy|Strong Sell|Buy|Hold|Sell)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def render_slips(docs):
    st.markdown('<div class="fr-catalog-label">◆ Cited passages</div>', unsafe_allow_html=True)
    with st.expander(f"Open filing drawer ({len(docs)} excerpts)"):
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "—")
            if page != "—" and page != "Unknown":
                page = page + 1 if isinstance(page, int) else page
            source = doc.metadata.get("source", "document")
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 480:
                snippet = snippet[:480].rsplit(" ", 1)[0] + " …"
            st.markdown(
                f"""<div class="fr-slip">
                <span class="fr-slip-num">{i:02d}</span>{snippet}
                <div class="fr-slip-meta">source: {source} · page {page}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ============================
# Session state init
# ============================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = load_existing_vectorstore()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ============================
# Sidebar
# ============================
with st.sidebar:
    st.markdown("## ① Upload financial reports")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs (10-K, annual report, earnings deck, etc.)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)

    if st.button("📥 Index documents", disabled=not uploaded_files):
        with st.spinner("Splitting, embedding, and filing the report(s)..."):
            n_chunks = index_files(uploaded_files, chunk_size, chunk_overlap)
        st.success(f"Indexed {n_chunks} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.indexed_files:
        st.caption("Filed this session: " + ", ".join(sorted(st.session_state.indexed_files)))

    st.markdown('<hr class="fr-rule">', unsafe_allow_html=True)
    st.markdown("## ② Analysis settings")
    model_name = st.selectbox("Model", ["mistral-small-2506", "mistral-large-latest"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    k = st.slider("Chunks returned (k)", 1, 30, 15)
    fetch_k = st.slider("Candidates scanned (fetch_k)", 10, 200, 60, step=10)
    lambda_mult = st.slider("Relevance ↔ Diversity", 0.0, 1.0, 0.7)

    st.markdown('<hr class="fr-rule">', unsafe_allow_html=True)
    st.caption(f"VECTOR STORE: `{PERSIST_DIR}`")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Reset archive"):
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)
            st.session_state.vectorstore_ready = False
            st.session_state.indexed_files = set()
            st.session_state.messages = []
            st.rerun()

# ============================
# Header
# ============================
st.markdown(
    """
<div class="fr-header">
    <div class="fr-eyebrow">Equity research · grounded in your filings only</div>
    <p class="fr-title">AI Financial Advisor</p>
    <p class="fr-sub">Upload financial reports, then ask about revenue, profitability, debt, risk, or get an investment rating — every answer is backed by cited pages.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state.vectorstore_ready:
    st.markdown(
        '<div class="fr-empty">No reports on file yet.<br>Upload one or more PDFs in the sidebar and click '
        '<b>Index documents</b> to begin.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# Suggested questions
st.markdown("**Suggested questions**")
cols = st.columns(3)
for i, q in enumerate(SUGGESTED_QUESTIONS):
    if cols[i % 3].button(q, key=f"sugg_{i}", use_container_width=True):
        st.session_state.pending_query = q

try:
    retriever = get_retriever(k, fetch_k, lambda_mult)
    llm = get_llm(model_name, temperature)
except Exception as e:
    st.error(f"Could not reach the model or vector store: {e}")
    st.stop()

# ============================
# Render chat history
# ============================
for msg in st.session_state.messages:
    avatar = "🗣️" if msg["role"] == "user" else "📊"
    with st.chat_message(msg["role"], avatar=avatar):
        rating = msg.get("rating")
        if rating:
            color = RATING_COLORS.get(rating, "#B08D57")
            st.markdown(
                f'<div class="fr-rating" style="background:{color};">Investment Rating: {rating.title()}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_slips(msg["sources"])

# ============================
# Chat input (typed or suggested)
# ============================
typed_query = st.chat_input("Ask about revenue, debt, risk, outlook, or request an investment rating...")
query = st.session_state.pending_query or typed_query
st.session_state.pending_query = None

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🗣️"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("Reviewing the filings..."):
            try:
                answer, docs = answer_query(query, retriever, llm)
            except Exception as e:
                answer, docs = f"The analysis could not be completed: {e}", []
        rating = extract_rating(answer)
        if rating:
            color = RATING_COLORS.get(rating, "#B08D57")
            st.markdown(
                f'<div class="fr-rating" style="background:{color};">Investment Rating: {rating.title()}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(answer)
        if docs:
            render_slips(docs)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": docs, "rating": rating}
    )
