import os
import re
import shutil
import tempfile

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="NewsDesk AI", page_icon="📰", layout="wide")

# Use a directory under the OS temp dir rather than a relative path, since
# some hosting environments (containers, certain PaaS setups) mount the
# app's working directory read-only and only allow writes under /tmp.
PERSIST_DIR = os.path.join(tempfile.gettempdir(), "newsdesk_chroma_db")

# ============================================================================
# Backend news source
# ----------------------------------------------------------------------------
# This API returns a fixed batch of the ~100 latest articles across ALL topics
# and sources (BBC, TechCrunch, Al Jazeera, Marketaux, Finnhub, SEC EDGAR,
# Reddit, Hacker News, etc.) — query params like ?q= or ?limit= are ignored.
# So we fetch the whole batch once and filter client-side by company name
# against title / description (NOT source/keywords/category — see
# filter_articles_for_company for why).
# ============================================================================
NEWS_API_URL = "https://news-pipeline-iqtb.onrender.com/api/articles"

SUGGESTED_QUESTIONS = [
    "Summarize the latest news.",
    "Any major risks or controversies?",
    "What are analysts saying?",
    "Recent product or business developments?",
    "Any regulatory or legal news?",
    "Overall sentiment from recent coverage?",
]


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
You are an expert Financial News Analyst.
Answer ONLY using the provided context, which is made up of recent news articles.
If the answer is not present in the context, reply:
"I could not find this information in the fetched news articles."
Whenever appropriate, analyze:
• Recent developments
• Market and analyst sentiment
• Risks or controversies
• Regulatory or legal news
• Competitive or industry context
Explain every conclusion using evidence from the articles.
Mention the article title/source whenever available.
Never use outside knowledge beyond the provided articles.
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
# STYLE — "NewsDesk": ink/paper archive language, newsroom-flavored
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

.nd-header {
    border: 1px solid rgba(176,141,87,0.4); border-radius: 2px; padding: 28px 32px;
    margin-bottom: 24px; background: var(--ink-soft); position: relative;
}
.nd-header::before {
    content: ""; position: absolute; inset: 6px; border: 1px solid rgba(176,141,87,0.25); pointer-events: none;
}
.nd-eyebrow {
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.18em; text-transform: uppercase;
    font-size: 11px; color: var(--brass); margin-bottom: 6px;
}
.nd-title { font-family: 'Lora', serif; font-weight: 600; font-size: 32px; color: var(--chalk); margin: 0; }
.nd-sub { font-size: 14px; color: rgba(242,239,230,0.55); margin-top: 6px; }

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

.nd-catalog-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--moss); margin: 10px 0 6px 2px;
}
.nd-slip {
    background: var(--paper); border: 1px solid var(--paper-dim); border-left: 3px solid var(--brass);
    border-radius: 2px; padding: 12px 14px; margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: var(--ink); line-height: 1.55;
}
.nd-slip-num {
    display: inline-block; background: var(--brass); color: var(--ink); font-weight: 700;
    font-size: 11px; padding: 1px 7px; border-radius: 2px; margin-right: 8px;
}
.nd-slip-meta { color: var(--moss); font-size: 11px; margin-top: 6px; opacity: 0.85; }
.nd-slip-meta a { color: var(--moss); }

.nd-article-row {
    background: var(--paper); border: 1px solid var(--paper-dim); border-radius: 3px;
    padding: 10px 14px; margin-bottom: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--ink);
}
.nd-article-row a { color: var(--ink); font-weight: 600; text-decoration: none; }
.nd-article-row .nd-meta { color: var(--moss); font-size: 11px; }

div[data-testid="stExpander"] { background: transparent; border: 1px solid rgba(176,141,87,0.35); border-radius: 3px; }
div[data-testid="stExpander"] summary { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--brass); }

div[data-testid="stChatInput"] { background: var(--ink-soft); border: 1px solid rgba(176,141,87,0.4); border-radius: 4px; }
div[data-testid="stChatInput"] textarea { color: var(--chalk) !important; }

.nd-rule { border: none; border-top: 1px dashed rgba(176,141,87,0.3); margin: 16px 0; }
.nd-empty {
    border: 1px dashed rgba(176,141,87,0.4); border-radius: 4px; padding: 40px 24px;
    text-align: center; color: rgba(242,239,230,0.6); font-family: 'Lora', serif; font-size: 17px; margin-top: 10px;
}
.nd-warn {
    border: 1px solid rgba(176,141,87,0.4); border-radius: 4px; padding: 14px 18px;
    color: rgba(242,239,230,0.75); font-size: 13px; margin-top: 10px; background: rgba(176,141,87,0.06);
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


def fetch_all_articles():
    """Pull the fixed batch of latest articles from the backend API."""
    resp = requests.get(NEWS_API_URL, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("data", [])


def clean_html(text):
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)


# ============================
# Single-company input validation
# ============================
_MULTI_COMPANY_SEPARATORS = re.compile(r"\s*(?:,|;|/|\band\b|\bor\b|&)\s*", re.IGNORECASE)


def is_single_company_name(raw_input):
    """Reject input that looks like more than one company name, so an
    analysis run is always scoped to exactly one company."""
    cleaned = raw_input.strip()
    if not cleaned:
        return False
    parts = [p for p in _MULTI_COMPANY_SEPARATORS.split(cleaned) if p.strip()]
    return len(parts) == 1


def filter_articles_for_company(articles, company_name):
    """
    Strict client-side filter: keep only articles that are genuinely about
    the given company, not articles that merely contain the company name as
    an incidental substring somewhere in the payload.

    Compared to a naive substring check, this:
      - Uses whole-word, case-insensitive regex matching (\\b...\\b), so a
        company like "Ford" won't match "Oxford", "Meta" won't match
        "Metadata", etc.
      - Only searches the title and description — the fields that actually
        describe what the article is about. The `source` field (e.g. "BBC",
        "TechCrunch") and generic `keywords`/`category` tags are deliberately
        excluded, since a match there is not reliable evidence the article
        is actually about the company (it can be an unrelated article that
        happens to share a tag, or a news outlet whose name overlaps).
    """
    needle = company_name.strip()
    if not needle:
        return []

    pattern = re.compile(r"\b" + re.escape(needle) + r"\b", re.IGNORECASE)

    matches = []
    for a in articles:
        title = a.get("title", "") or ""
        description = a.get("description", "") or ""

        if pattern.search(title) or pattern.search(description):
            matches.append(a)

    return matches


def build_news_documents(articles, company_name):
    docs = []
    for a in articles:
        title = a.get("title", "Untitled")
        description = a.get("description", "") or ""
        content = clean_html(a.get("content", "") or "")
        keywords = ", ".join(a.get("keywords", []) or [])
        body_parts = [p for p in [description, content] if p]
        body = "\n\n".join(body_parts) if body_parts else "(no article text available)"
        text = f"{title}\n\n{body}\n\nKeywords: {keywords}"
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "title": title,
                    "source": a.get("url", ""),
                    "source_name": a.get("source", "Unknown"),
                    "published": a.get("published_at", "Unknown date"),
                    "sentiment_polarity": (a.get("sentiment") or {}).get("polarity"),
                    "company": company_name,
                },
            )
        )
    return docs


def _wipe_persist_dir():
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)


def _write_chunks_to_chroma(chunks, retry_on_readonly=True):
    """Create a fresh Chroma store at PERSIST_DIR and write chunks to it.
    If the on-disk DB is stuck in a readonly/corrupted state (e.g. leftover
    lock/file from a crashed run, or a permissions problem), wipe the
    persist dir once more and retry before giving up with a clear error."""
    embedding_model = get_embedding_model()
    try:
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
        vectorstore.add_documents(chunks)
    except Exception as e:
        is_readonly_error = "readonly database" in str(e).lower() or "code: 1032" in str(e)
        if retry_on_readonly and is_readonly_error:
            _wipe_persist_dir()
            _write_chunks_to_chroma(chunks, retry_on_readonly=False)
        else:
            raise RuntimeError(
                f"Vector store write failed: {e}. If this keeps happening, the app's "
                f"working directory may not be writable — check permissions on "
                f"'{PERSIST_DIR}' (or its parent folder) or free up disk space."
            ) from e


def index_company_news(company_name, chunk_size, chunk_overlap):
    all_articles = fetch_all_articles()
    matches = filter_articles_for_company(all_articles, company_name)
    if not matches:
        return 0, [], len(all_articles), None

    docs = build_news_documents(matches, company_name)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # This app analyzes ONE company at a time. Wipe any previously indexed
    # company's chunks before adding the new ones, otherwise Chroma just
    # accumulates documents across runs and the retriever can pull back
    # unrelated articles from a company indexed earlier in the session.
    _wipe_persist_dir()

    try:
        _write_chunks_to_chroma(chunks)
    except Exception as e:
        return 0, matches, len(all_articles), str(e)

    st.session_state.vectorstore_ready = True
    st.session_state.indexed_company = company_name
    st.session_state.last_articles = matches
    return len(chunks), matches, len(all_articles), None


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


def get_retriever(k, fetch_k, lambda_mult, company_name=None):
    embedding_model = get_embedding_model()
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    search_kwargs = {"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    if company_name:
        # Defense in depth: even though the store is wiped per company on
        # index, this guarantees retrieval never crosses into another
        # company's chunks if that invariant is ever broken.
        search_kwargs["filter"] = {"company": company_name}
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )


def build_context(docs):
    context = ""
    for doc in docs:
        title = doc.metadata.get("title", "Untitled")
        source_name = doc.metadata.get("source_name", "Unknown")
        published = doc.metadata.get("published", "Unknown date")
        context += f"\n\n========== {title} ({source_name}, {published}) ==========\n"
        context += doc.page_content
    return context


def answer_query(query, retriever, llm):
    docs = retriever.invoke(query)
    if not docs:
        return "I could not find this information in the fetched news articles.", []
    context = build_context(docs)
    final_prompt = PROMPT.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)
    return response.content, docs


def render_slips(docs):
    st.markdown('<div class="nd-catalog-label">◆ Cited articles</div>', unsafe_allow_html=True)
    with st.expander(f"Open clippings drawer ({len(docs)} excerpts)"):
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Untitled")
            link = doc.metadata.get("source", "")
            source_name = doc.metadata.get("source_name", "Unknown")
            published = doc.metadata.get("published", "Unknown date")
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 480:
                snippet = snippet[:480].rsplit(" ", 1)[0] + " …"
            link_html = f'<a href="{link}" target="_blank">{title}</a>' if link else title
            st.markdown(
                f"""<div class="nd-slip">
                <span class="nd-slip-num">{i:02d}</span>{snippet}
                <div class="nd-slip-meta">{link_html} · {source_name} · {published}</div>
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
if "indexed_company" not in st.session_state:
    st.session_state.indexed_company = None
if "last_articles" not in st.session_state:
    st.session_state.last_articles = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ============================
# Sidebar
# ============================
with st.sidebar:
    st.markdown("## ① Fetch company news")
    company_name = st.text_input(
        "Company name",
        placeholder="e.g. Tesla, Duolingo, Microsoft",
        help="Enter exactly ONE company. The analysis will be scoped strictly to that company's articles.",
    )

    company_name_clean = company_name.strip()
    company_valid = bool(company_name_clean) and is_single_company_name(company_name_clean)
    if company_name_clean and not company_valid:
        st.warning(
            "Please enter only **one** company name (no commas, '&', 'and'/'or'). "
            "Analysis is scoped to a single company at a time."
        )

    chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=800, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=150, step=50)

    if st.button("📡 Fetch & index news", disabled=not company_valid):
        with st.spinner(f"Pulling the latest feed and filtering for '{company_name_clean}'..."):
            try:
                n_chunks, matches, total_fetched, err = index_company_news(
                    company_name_clean, chunk_size, chunk_overlap
                )
            except Exception as e:
                # Fetch itself failed (network / API) before we ever got articles.
                n_chunks, matches, total_fetched, err = 0, [], 0, f"Could not reach the news API: {e}"

        if n_chunks:
            st.success(f"Indexed {n_chunks} chunks from {len(matches)} matching article(s) (out of {total_fetched} fetched).")
        elif err:
            # A real failure occurred (fetch or vector-store write) — don't
            # also claim "no articles found", that would be misleading.
            st.error(err)
        elif total_fetched:
            st.warning(
                f"No articles specifically about '{company_name_clean}' were found in the latest {total_fetched} articles. "
                "The feed only covers recent general news, not a full historical company search."
            )

    if st.session_state.indexed_company:
        st.caption(f"Currently indexed: **{st.session_state.indexed_company}**")

    st.markdown('<hr class="nd-rule">', unsafe_allow_html=True)
    st.markdown("## ② Analysis settings")
    model_name = st.selectbox("Model", ["mistral-small-2506", "mistral-large-latest"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    k = st.slider("Chunks returned (k)", 1, 30, 10)
    fetch_k = st.slider("Candidates scanned (fetch_k)", 10, 200, 40, step=10)
    lambda_mult = st.slider("Relevance ↔ Diversity", 0.0, 1.0, 0.7)

    st.markdown('<hr class="nd-rule">', unsafe_allow_html=True)
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
            st.session_state.indexed_company = None
            st.session_state.last_articles = []
            st.session_state.messages = []
            st.rerun()

# ============================
# Header
# ============================
st.markdown(
    """
<div class="nd-header">
    <div class="nd-eyebrow">Live news · grounded in fetched articles only</div>
    <p class="nd-title">NewsDesk AI</p>
    <p class="nd-sub">Type a company name to pull matching articles from the live feed, then ask questions — every answer is backed by cited articles, no outside knowledge.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="nd-warn">Note: the news feed only returns the latest ~100 articles across all topics — '
    "it's not a full historical search. If a company hasn't been in recent headlines, no matches will be found. "
    "Analysis is always scoped to exactly one company at a time.</div>",
    unsafe_allow_html=True,
)

if not st.session_state.vectorstore_ready:
    st.markdown(
        '<div class="nd-empty">No news indexed yet.<br>Type a single company name in the sidebar and click '
        '<b>Fetch &amp; index news</b> to begin.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if st.session_state.last_articles:
    with st.expander(f"📰 Articles matched for {st.session_state.indexed_company} ({len(st.session_state.last_articles)})"):
        for a in st.session_state.last_articles:
            sentiment = (a.get("sentiment") or {}).get("polarity", 0.0)
            st.markdown(
                f"""<div class="nd-article-row">
                <a href="{a.get('url', '')}" target="_blank">{a.get('title', 'Untitled')}</a><br>
                <span class="nd-meta">{a.get('source', 'Unknown')} · {a.get('published_at', 'Unknown date')} · sentiment: {sentiment:+.2f}</span>
                </div>""",
                unsafe_allow_html=True,
            )

st.markdown("**Suggested questions**")
cols = st.columns(3)
for i, q in enumerate(SUGGESTED_QUESTIONS):
    if cols[i % 3].button(q, key=f"sugg_{i}", use_container_width=True):
        st.session_state.pending_query = q

try:
    retriever = get_retriever(k, fetch_k, lambda_mult, st.session_state.indexed_company)
    llm = get_llm(model_name, temperature)
except Exception as e:
    st.error(f"Could not reach the model or vector store: {e}")
    st.stop()

# ============================
# Render chat history
# ============================
for msg in st.session_state.messages:
    avatar = "🗣️" if msg["role"] == "user" else "📰"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_slips(msg["sources"])

# ============================
# Chat input (typed or suggested)
# ============================
typed_query = st.chat_input("Ask about recent news, sentiment, risks, or developments...")
query = st.session_state.pending_query or typed_query
st.session_state.pending_query = None

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🗣️"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="📰"):
        with st.spinner("Reviewing the clippings..."):
            try:
                answer, docs = answer_query(query, retriever, llm)
            except Exception as e:
                answer, docs = f"The analysis could not be completed: {e}", []
        st.markdown(answer)
        if docs:
            render_slips(docs)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": docs})
