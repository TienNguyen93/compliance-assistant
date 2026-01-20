"""
Streamlit UI for Compliance Assistant
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.llm_agent import ComplianceAgent
import config
import psutil

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
def load_agent():
    """Load agent once and cache it"""
    try:
        return ComplianceAgent(model="llama3.1")
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.info("Make sure Ollama is installed and running. Run: `ollama pull llama3.1`")
        st.stop()

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
    # Sidebar
    with st.sidebar:
        st.header("FDA Compliance Assistant")
        st.header("Settings")
        
        # Model selection
        model_choice = st.selectbox(
            "Ollama Model",
            ["llama3.1"],
            help="Select the Ollama model to use"
        )
        
        # Number of sources
        top_k = st.slider(
            "Number of sources",
            min_value=1,
            max_value=10,
            value=5,
            help="How many document chunks to retrieve for context"
        )
        
        # Show sources toggle
        show_sources = st.checkbox("Show source documents", value=True)
        show_context = st.checkbox("Show retrieved context", value=False)

        
        st.markdown("---")
        
        # Sources section
        st.header("Sources")
        try:
            from pathlib import Path
            import base64
            fda_guidance_path = Path(__file__).resolve().parent / "data" / "raw" / "fda_guidance"
            pdf_files = sorted([f for f in fda_guidance_path.glob("*.pdf")])
            
            if pdf_files:
                for pdf_path in pdf_files:
                    display_name = pdf_path.name.replace("-", " ")
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_base64 = base64.b64encode(pdf_file.read()).decode('utf-8')
                        st.markdown(f"<a href='data:application/pdf;base64,{pdf_base64}' target='_blank' style='text-decoration:none'>📄 {display_name}</a>", unsafe_allow_html=True)
            else:
                st.write("No PDFs found in fda_guidance folder")
        except Exception as e:
            st.error(f"Error loading sources: {e}")
        
        st.markdown("---")
        
        # TODO: Add real usage stats
        # Usage statistics
        st.header("Usage Statistics")

        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)

        # # RAM Usage
        # memory = psutil.virtual_memory()
        # ram_available = memory.available / (1024**3)  # Convert to GB
        # ram_total = memory.total / (1024**3)  # Convert to GB
        # ram_usage_percent = memory.percent  

        # # GPU Usage (if available)
        # gpu_usage = "N/A"
        # gpu_memory_used = "N/A"
        # gpu_memory_total = "N/A"
        # try:
        #     pynvml.nvmlInit()
        #     device_count = pynvml.nvmlDeviceGetCount()
        #     if device_count > 0:
        #         handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get first GPU
        #         utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        #         memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        #         gpu_usage = utilization.gpu  # GPU core utilization (%)
        #         gpu_memory_used = memory_info.used / (1024**3)  # Convert to GB
        #         gpu_memory_total = memory_info.total / (1024**3)  # Convert to GB
        #     pynvml.nvmlShutdown()
        # except (pynvml.NVMLError, ImportError):
        #     pass  # GPU metrics will remain "N/A" if pynvml is not available or no GPU is detected

        # try:
        #     agent = load_agent()
        #     stats = agent.retriever.get_stats()
            
        # except Exception as e:
        #     st.error(f"Error: {e}")

        
        st.markdown("---")
        
        # Sample queries
        st.header("Sample Queries")
        sample_queries = [
            "What does terminal sterilization usually involve?",
            "Which guidance is under section 503B of the Federal Food, Drug, and Cosmetic Act?",
            "What should manufacturers of reusable devices consider?",
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{query}", use_container_width=True):
                st.session_state.sample_query = query
    
    # Main content
    col1, col2 = st.columns([4, 1])
    
    with col1:
        default_query = st.session_state.get('sample_query', '')
        query = st.text_area(
            "Ask a question about FDA guidance:",
            value=default_query,
            height=100,
            placeholder="e.g., What does terminal sterilization usually involve?",
            key="query_input"
        )
        
        if 'sample_query' in st.session_state:
            del st.session_state.sample_query
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button("🤖 Generate Answer", type="primary", use_container_width=True)
    
    # Process query
    if search_button and query:
        with st.spinner("🔍 Retrieving relevant documents..."):
            try:
                agent = load_agent()
                
                # Progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Searching vector database...")
                progress_bar.progress(33)
                
                status_text.text("Generating answer with Llama 3.1...")
                progress_bar.progress(66)
                
                # Generate answer
                result = agent.generate_answer(query, top_k=top_k)
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                # Clear progress indicators
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Display answer
                st.markdown("---")
                st.markdown("### 💡 Answer")
                
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Display sources
                if show_sources:
                    st.markdown("---")
                    st.markdown(f"### 📚 Sources ({result['num_sources']} documents used)")
                    
                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(
                            f"📄 Source {i}: {source['metadata']['filename']} "
                            f"(Relevance: {source['relevance_score']:.1%})"
                        ):
                            st.markdown(f"**Category:** {source['metadata']['category']}")
                            st.markdown(f"**Relevance Score:** {source['relevance_score']:.1%}")
                            st.markdown("**Content:**")
                            st.write(source['text'])
                
                # Show context (debug mode)
                if show_context:
                    with st.expander("🔍 View Retrieved Context (Debug)"):
                        st.text(result['context'])
                
                # Export options
                st.markdown("---")
                col_a, col_b, col_c = st.columns([1, 1, 2])
                
                with col_a:
                    if st.button("💾 Export as JSON"):
                        import json
                        json_str = json.dumps({
                            'question': result['question'],
                            'answer': result['answer'],
                            'sources': [
                                {
                                    'filename': s['metadata']['filename'],
                                    'relevance': s['relevance_score'],
                                    'text': s['text']
                                }
                                for s in result['sources']
                            ]
                        }, indent=2)
                        
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"answer_{query[:30]}.json",
                            mime="application/json"
                        )
                
                with col_b:
                    if st.button("📋 Copy Answer"):
                        st.code(result['answer'], language=None)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)
    
    elif search_button and not query:
        st.warning("⚠️ Please enter a question to get started.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with Streamlit • LangChain • ChromaDB • Ollama (Llama 3.1)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()