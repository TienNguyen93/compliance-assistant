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
# import psutil

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

def initialize_session_state():
    """Initialize session state for chat history"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def add_to_chat_history(question, answer, sources):
    """Add a Q&A pair to chat history"""
    st.session_state.chat_history.append({
        'question': question,
        'answer': answer,
        'sources': sources,
        'num_sources': len(sources)
    })

def main():
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("FDA Compliance Assistant")
        st.header("Settings")
        
        st.text("Ollama Model: Llama3.1")
        
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

        
        # st.markdown("---")
        
        # # Sources section
        # st.header("Sources")
        # try:
        #     from pathlib import Path
        #     import base64
        #     fda_guidance_path = Path(__file__).resolve().parent / "data" / "raw" / "fda_guidance"
        #     pdf_files = sorted([f for f in fda_guidance_path.glob("*.pdf")])
            
        #     if pdf_files:
        #         for pdf_path in pdf_files:
        #             display_name = pdf_path.name.replace("-", " ")
        #             with open(pdf_path, "rb") as pdf_file:
        #                 pdf_base64 = base64.b64encode(pdf_file.read()).decode('utf-8')
        #                 st.markdown(f"<a href='data:application/pdf;base64,{pdf_base64}' target='_blank' style='text-decoration:none'>📄 {display_name}</a>", unsafe_allow_html=True)
        #     else:
        #         st.write("No PDFs found in fda_guidance folder")
        # except Exception as e:
        #     st.error(f"Error loading sources: {e}")
        
        # TODO: Add real usage stats
        # Usage statistics
        # st.header("Usage Statistics")

        # # CPU usage
        # cpu_usage = psutil.cpu_percent(interval=1)

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

        
        # st.markdown("---")
        
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
        
        st.markdown("---")
        
        # Chat History in Sidebar
        if st.session_state.chat_history:
            st.header("Chat History")
            if st.button("Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.selected_chat = None
                st.rerun()
            
            for i, chat in enumerate(st.session_state.chat_history):
                if st.button(f"Q: {chat['question'][:50]}...", key=f"chat_{i}", use_container_width=True):
                    st.session_state.selected_chat = i
                    st.rerun()
    
    # Initialize selected_chat if not present
    if 'selected_chat' not in st.session_state:
        st.session_state.selected_chat = None
    
    # Main content
    st.header("Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load selected chat from history if clicked
    if st.session_state.selected_chat is not None:
        selected_idx = st.session_state.selected_chat
        if selected_idx < len(st.session_state.chat_history):
            selected_chat_data = st.session_state.chat_history[selected_idx]
            st.markdown(f"### Question: {selected_chat_data['question']}")
            st.markdown(f"**Answer:**")
            st.markdown(selected_chat_data['answer'])
            if selected_chat_data['sources']:
                st.markdown(f"*Relevant documents: {selected_chat_data['num_sources']}*")
            st.markdown("---")

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Check if a sample query was clicked and automatically submit it
    if 'sample_query' in st.session_state:
        sample_query = st.session_state.sample_query
        del st.session_state.sample_query
        
        # Store and display the sample query
        st.session_state.messages.append({"role": "user", "content": sample_query})
        with st.chat_message("user"):
            st.markdown(sample_query)

        # Generate a response using the ComplianceAgent.
        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving relevant documents..."):
                try:
                    agent = load_agent()
                    
                    # Generate answer
                    result = agent.generate_answer(sample_query, top_k=top_k)
                    
                    # Display the answer
                    st.markdown(result["answer"])
                    
                    # Add to chat history
                    add_to_chat_history(sample_query, result["answer"], result['sources'])
                    
                    # Store in messages
                    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    
    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("Start typing..."):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the ComplianceAgent.
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant documents..."):
                try:
                    agent = load_agent()
                    
                    # Generate answer
                    result = agent.generate_answer(prompt, top_k=top_k)
                    
                    # Display the answer
                    st.markdown(result["answer"])
                    
                    # Add to chat history
                    add_to_chat_history(prompt, result["answer"], result['sources'])
                    
                    # Store in messages
                    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()