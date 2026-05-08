"""
DocuMind AI — Streamlit Chat Interface
Production-grade UI for the agentic RAG system
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Fix .env path
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.append(os.path.dirname(__file__))

from retrieval import ask
from ingestion import ingest_pdf
import tempfile

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E40AF;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .source-card {
        background: #F3F4F6;
        border-left: 3px solid #1E40AF;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
    }
    .metric-card {
        background: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 DocuMind AI")
    st.markdown("*Agentic RAG Document Intelligence*")
    st.divider()

    # Document upload
    st.markdown("#### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF to query",
        type=["pdf"],
        help="Upload any PDF — resume, report, or document"
    )

    if uploaded_file:
        if st.button("🚀 Ingest Document", type="primary"):
            with st.spinner("Ingesting document..."):
                try:
                    # Save to temp file and ingest
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    ingest_pdf(tmp_path)
                    os.unlink(tmp_path)

                    st.success(
                        f"✅ '{uploaded_file.name}' ingested!"
                    )
                    st.info(
                        "Document is now searchable. "
                        "Ask questions below!"
                    )
                except Exception as e:
                    st.error(f"Ingestion failed: {str(e)}")

    st.divider()

    # System info
    st.markdown("#### ⚙️ System")
    st.markdown("""
    - **LLM:** GPT-4o (Azure OpenAI)
    - **Embeddings:** text-embedding-3-small
    - **Vector DB:** Pinecone
    - **Reranker:** Cohere
    - **Framework:** LangGraph
    """)

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<div style='text-align:center;color:#9CA3AF;"
        "font-size:0.75rem'>Built with LangGraph + Azure OpenAI"
        "</div>",
        unsafe_allow_html=True
    )


# ── Main area ─────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🧠 DocuMind AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">'
    'Agentic RAG · Azure OpenAI · Pinecone · LangGraph'
    '</div>',
    unsafe_allow_html=True
)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("LLM", "GPT-4o", "Azure")
with col2:
    st.metric("Vector DB", "Pinecone", "Serverless")
with col3:
    st.metric("Reranker", "Cohere", "v3.0")
with col4:
    st.metric("Agents", "4", "LangGraph")

st.divider()

# ── Chat interface ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "👋 Hello! I'm **DocuMind AI** — an agentic RAG "
            "assistant.\n\n"
            "I can answer questions about any document you upload. "
            "A resume has already been loaded — try asking:\n"
            "- *What are the candidate's GenAI skills?*\n"
            "- *What projects has the candidate built?*\n"
            "- *How many years of experience do they have?*"
        ),
        "sources": []
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources if available
        if msg.get("sources"):
            with st.expander(
                f"📚 Sources ({len(msg['sources'])})",
                expanded=False
            ):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>Source {i+1}</b> | '
                        f'{src["source"]} | '
                        f'Page {src["page"]} | '
                        f'Score: {src["score"]:.3f}<br>'
                        f'<i>{src["text"][:120]}...</i>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# Chat input
if prompt := st.chat_input("Ask about the document..."):

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "sources": []
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching → 🔄 Reranking → ✍️ Generating..."):
            try:
                result  = ask(prompt, verbose=False)
                answer  = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                # Show sources
                if sources:
                    with st.expander(
                        f"📚 Sources ({len(sources)})",
                        expanded=False
                    ):
                        for i, src in enumerate(sources):
                            st.markdown(
                                f'<div class="source-card">'
                                f'<b>Source {i+1}</b> | '
                                f'{src["source"]} | '
                                f'Page {src["page"]} | '
                                f'Score: {src["score"]:.3f}<br>'
                                f'<i>{src["text"][:120]}...</i>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                # Save to history
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": error_msg,
                    "sources": []
                })