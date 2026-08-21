import streamlit as st
import os
from pathlib import Path

# Import your core pipeline components & Vector Store Class directly
from src.vector_store import FinancialVectorStore
from src.rag_pipeline import FinancialRAGPipeline

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FinRAG Analyst - Enterprise Financial Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. INITIALIZE VECTOR STORE & PIPELINE (Cached) ---
@st.cache_resource
def load_rag_system():
    try:
        # Initialize vector store using your existing FinancialVectorStore class
        vector_store_manager = FinancialVectorStore()
        vector_store = vector_store_manager.get_vectorstore()
        
        # Initialize RAG Pipeline
        pipeline = FinancialRAGPipeline(vector_store)
        return pipeline
    except Exception as e:
        st.error(f"❌ Failed to initialize Vector Store / RAG Pipeline: {e}")
        return None

rag_pipeline = load_rag_system()

# --- 4. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=80)
    st.title("FinRAG Controls")
    st.markdown("---")
    
    st.info("💡 **Tip:** Use filters to target specific filings or leave them default for broad analysis.")
    
    # Ticker Filter Selector
    selected_ticker = st.selectbox(
        "🏢 Select Company (Ticker)",
        options=["All Companies", "MSFT", "TSLA", "JPM"],
        index=0
    )
    ticker_filter = None if selected_ticker == "All Companies" else selected_ticker

    # Year Filter Selector
    selected_year = st.selectbox(
        "📅 Select Filing Year",
        options=["All Years", "2025", "2024", "2023"],
        index=0
    )
    year_filter = None if selected_year == "All Years" else selected_year

    st.markdown("---")
    st.markdown("### 📊 Supported Filings")
    st.markdown("- **Microsoft (MSFT):** 10-K & Transcripts")
    st.markdown("- **Tesla (TSLA):** 10-K & Transcripts")
    st.markdown("- **JPMorgan (JPM):** 10-K & Transcripts")

    st.markdown("---")
    if st.button("🧹 Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()

# --- 5. MAIN CHAT INTERFACE ---
st.title("🤖 FinRAG Enterprise Financial Assistant")
st.markdown("Analyze SEC 10-K filings and Q4 earnings transcripts instantly with **zero-hallucination guardrails**.")

# Initialize chat history state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your FinRAG Analyst. Ask me any question regarding Microsoft, Tesla, or JPMorgan Chase's financial reports."
        }
    ]

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. USER INPUT & QUERY EXECUTION ---
if prompt := st.chat_input("Ask a financial question (e.g., What was Tesla's 2024 revenue?)"):
    
    # Append user message to state & display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        if not rag_pipeline:
            st.error("⚠️ RAG Pipeline is not initialized. Please check your Qdrant connection.")
        else:
            with st.spinner("⏳ Analyzing financial documents & tables..."):
                try:
                    # Get the LCEL chain with dynamic sidebar filters applied
                    chain = rag_pipeline.get_chain(
                        ticker_filter=ticker_filter, 
                        year_filter=year_filter
                    )
                    
                    # Invoke chain with user question
                    response = chain.invoke(prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Save response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ An error occurred during execution: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})