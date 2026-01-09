"""
Streamlit UI for Compliance Assistant
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.retriever import ComplianceRetriever
import config

# Page configuration
st.set_page_config(
    page_title="FDA Compliance Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
    }
    .source-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .relevance-score {
        color: #28a745;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize retriever (cached for performance)
@st.cache_resource
def load_retriever():
    """Load retriever once and cache it"""
    try:
        return ComplianceAgent(model="llama3.1")
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.info("Make sure Ollama is installed and running. Run: `ollama pull llama3.1`")
        st.stop()

    # return ComplianceRetriever()

def format_relevance_score(score):
    """Format relevance score with color coding"""
    if score >= 0.8:
        color = "#28a745"  # Green
        label = "High"
    elif score >= 0.6:
        color = "#ffc107"  # Yellow
        label = "Medium"
    else:
        color = "#dc3545"  # Red
        label = "Low"
    
    return f'<span style="color: {color}; font-weight: bold;">{score:.1%} ({label})</span>'

def main():
    # Header
    st.markdown('<p class="main-header">📋 FDA Compliance Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Intelligent document retrieval for FDA guidance documents</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Number of results
        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=5,
            help="How many document chunks to retrieve"
        )
        
        # Show metadata toggle
        show_metadata = st.checkbox("Show detailed metadata", value=True)
        
        st.markdown("---")
        
        # System info
        st.subheader("📊 System Info")
        
        try:
            retriever = load_retriever()
            stats = retriever.get_stats()
            
            st.metric("Total Document Chunks", stats['total_chunks'])
            st.metric("Embedding Model", "MiniLM-L6-v2", delta="384 dimensions")
            
            # Categories
            if stats.get('categories'):
                st.write("**Document Categories:**")
                for cat in stats['categories']:
                    st.write(f"• {cat}")
        
        except Exception as e:
            st.error(f"Error loading system info: {e}")
        
        st.markdown("---")
        
        # Sample queries
        st.subheader("💡 Sample Queries")
        sample_queries = [
            "What are cleaning validation requirements?",
            "How should sterile products be manufactured?",
            "What is process validation?",
            "What are critical control points in manufacturing?",
        ]
        
        for query in sample_queries:
            if st.button(query, key=query, use_container_width=True):
                st.session_state.sample_query = query
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        default_query = st.session_state.get('sample_query', '')
        query = st.text_input(
            "Ask a question about FDA guidance:",
            value=default_query,
            placeholder="e.g., What are the requirements for cleaning validation?",
            key="query_input"
        )
        
        # Clear sample query after use
        if 'sample_query' in st.session_state:
            del st.session_state.sample_query
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Process query
    if search_button and query:
        with st.spinner("Searching FDA guidance documents..."):
            try:
                retriever = load_retriever()
                results = retriever.search(query, top_k=top_k)
                
                # Display results count
                st.success(f"Found {results['num_results']} relevant document chunks")
                
                # Display results
                st.markdown("---")
                st.subheader("📄 Search Results")
                
                for i, result in enumerate(results['results'], 1):
                    with st.container():
                        # Result card
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h3 style="margin: 0;">Result {i}</h3>
                                <span class="relevance-score">Relevance: {format_relevance_score(result['relevance_score'])}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Source information
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.markdown(f"**📁 Source:** {result['metadata']['filename']}")
                        
                        with col_b:
                            st.markdown(f"**📂 Category:** {result['metadata']['category']}")
                        
                        # Document text
                        st.markdown("**Content:**")
                        st.write(result['text'])
                        
                        # Metadata (optional)
                        if show_metadata:
                            with st.expander("🔍 View Metadata"):
                                st.json(result['metadata'])
                        
                        st.markdown("---")
                
                # Export option
                if st.button("💾 Export Results as JSON"):
                    import json
                    json_str = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"search_results_{query[:30]}.json",
                        mime="application/json"
                    )
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)
    
    elif search_button and not query:
        st.warning("Please enter a question to search.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>FDA Compliance Assistant</strong> | Built with Streamlit, LangChain, and ChromaDB</p>
        <p>Powered by sentence-transformers for semantic search</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()