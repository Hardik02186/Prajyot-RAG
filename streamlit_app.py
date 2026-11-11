import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        border-left: 4px solid #17a2b8;
    }
    .metric-box {
        background-color: #fff3cd;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8001"

# Sidebar configuration
with st.sidebar:
    st.markdown("### Configuration")
    
    # API URL configuration
    api_url = st.text_input(
        "API URL",
        value=st.session_state.api_url,
        help="FastAPI server URL"
    )
    st.session_state.api_url = api_url
    
    # Mode selection
    mode = st.radio(
        "Response Mode",
        ["Normal (Full Quality)", "⚡ Fast (Quick Response)"],
        help="Normal: Uses reranking for better quality\nFast: Skips reranking for faster response"
    )
    
    # Settings
    st.markdown("###  Settings")
    show_metrics = st.checkbox("Show Response Metrics", value=True)
    auto_scroll = st.checkbox("Auto-scroll to Latest", value=True)
    
    # Connection status
    st.markdown("### Server Status")
    try:
        response = requests.get(f"{st.session_state.api_url}/health", timeout=2)
        if response.status_code == 200:
            st.success("Connected")
            
            # Show stats
            try:
                stats_response = requests.get(f"{st.session_state.api_url}/stats", timeout=5)
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    st.info(f"""
**Vector Store Stats:**
- Total Vectors: {stats['total_vectors']}
- Chunks: {stats['total_chunks']}
- Files Indexed: {stats['processed_files']}
                    """)
            except:
                pass
        else:
            st.error("Connection Failed")
    except requests.exceptions.ConnectionError:
        st.error("cannot reach server")
    except requests.exceptions.Timeout:
        st.warning("⚠️ Server timeout")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
    
    # Debug mode
    if st.checkbox("Debug Mode"):
        st.markdown("### Debug Tools")
        if st.button("Run Timing Test"):
            with st.spinner("Testing component speeds..."):
                try:
                    response = requests.get(f"{st.session_state.api_url}/debug/timing", timeout=30)
                    if response.status_code == 200:
                        timing = response.json()
                        st.json(timing)
                    else:
                        st.error("Failed to get timing data")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Main content
st.markdown('<div class="main-header">RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown("Ask questions about your documents and get instant answers powered by AI!")

# Question input section
st.markdown("### Ask a Question")
question = st.text_area(
    "Enter your question:",
    placeholder="e.g., What is the main purpose of this document?",
    height=100,
    label_visibility="collapsed"
)

# Button layout
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    submit_button = st.button("Get Answer", use_container_width=True)

with col2:
    fast_mode = st.button("Fast Answer", use_container_width=True)

with col3:
    if st.button("Paste", use_container_width=True):
        st.info("Paste your questions in the text area above!")

# Process question
if submit_button and question.strip():
    mode_endpoint = "chat/fast" if mode == "Fast (Quick Response)" else "chat"
    
    with st.spinner("Processing your question..."):
        start_time = time.time()
        try:
            response = requests.post(
                f"{st.session_state.api_url}/{mode_endpoint}",
                json={"question": question},
                timeout=150
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": question,
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "time": result.get("response_time_seconds", elapsed_time),
                    "mode": mode
                })
                
                # Display answer
                st.success("Answer Generated!")
                
                st.markdown(f'<div class="answer-box"><h3>Answer</h3><p>{result.get("answer", "")}</p></div>', unsafe_allow_html=True)
                
                # Display sources
                sources = result.get("sources", [])
                if sources:
                    st.markdown("### Sources")
                    for i, source in enumerate(sources, 1):
                        st.markdown(f'<div class="source-box"><b>{i}. {source.get("source", "Unknown")}</b> (Page {source.get("page", "?")})</div>', unsafe_allow_html=True)
                
                # Display metrics
                if show_metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Response Time", f"{result.get('response_time_seconds', elapsed_time):.2f}s")
                    with col2:
                        st.metric("Sources Used", len(sources))
                    with col3:
                        st.metric("Mode", mode.split()[0])
                
            else:
                st.error(f"Error: {response.status_code}")
                st.error(response.json().get("detail", "Unknown error"))
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. Server may be overloaded. Try Fast Mode!")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to server at {st.session_state.api_url}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif fast_mode and question.strip():
    with st.spinner("Fast processing..."):
        start_time = time.time()
        try:
            response = requests.post(
                f"{st.session_state.api_url}/chat/fast",
                json={"question": question},
                timeout=60
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": question,
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "time": result.get("response_time_seconds", elapsed_time),
                    "mode": "⚡ Fast"
                })
                
                st.success("⚡ Quick Answer Generated!")
                
                st.markdown(f'<div class="answer-box"><h3>Answer</h3><p>{result.get("answer", "")}</p></div>', unsafe_allow_html=True)
                
                sources = result.get("sources", [])
                if sources:
                    st.markdown("### Sources")
                    for i, source in enumerate(sources, 1):
                        st.markdown(f'<div class="source-box"><b>{i}. {source.get("source", "Unknown")}</b> (Page {source.get("page", "?")})</div>', unsafe_allow_html=True)
                
                if show_metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Response Time", f"{result.get('response_time_seconds', elapsed_time):.2f}s")
                    with col2:
                        st.metric("Sources Used", len(sources))
                    with col3:
                        st.metric("Mode", "⚡ Fast")
                
            else:
                st.error(f"Error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            st.error("Fast mode timeout. Try Normal mode!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Chat history section
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### Chat History")
    
    # Reverse to show latest first
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"**Q:** {chat['question'][:50]}... ({chat['timestamp']})", expanded=False):
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f"**Answer:** {chat['answer']}")
            
            if chat['sources']:
                st.markdown("**Sources:**")
                for source in chat['sources']:
                    st.caption(f"- {source.get('source', 'Unknown')} (Page {source.get('page', '?')})")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"{chat['time']:.2f}s")
            with col2:
                st.caption(f"{len(chat['sources'])} sources")
            with col3:
                st.caption(f"{chat['mode']}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    <p>Your questions are processed locally. No data is stored on external servers.</p>
    <p>For best results, make sure the FastAPI server is running on {}</p>
</div>
""".format(st.session_state.api_url), unsafe_allow_html=True)
