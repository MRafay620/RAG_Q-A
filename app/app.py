import os
import tempfile
import streamlit as st
from pathlib import Path
from io import BytesIO
import pandas as pd

# Import our main system
from main import DocumentAnalyzer, TextExtractor, TokenCounter

# Set page configuration
st.set_page_config(
    page_title="Dr. X's Publications Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "analyzer" not in st.session_state:
    st.session_state.analyzer = DocumentAnalyzer()
if "performance_metrics" not in st.session_state:
    st.session_state.performance_metrics = []

# Add CSS for better UI
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e6f7ff;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    .chat-message .user-avatar {
        background-color: #6c757d;
        color: white;
    }
    .chat-message .assistant-avatar {
        background-color: #0096c7;
        color: white;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary location and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def display_chat_messages():
    """Display the conversation history in a chat-like format."""
    for i, (role, text) in enumerate(st.session_state.conversation_history):
        if role == "user":
            with st.container():
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="avatar user-avatar">👤</div>
                    <div class="message">{text}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="avatar assistant-avatar">🤖</div>
                    <div class="message">{text}</div>
                </div>
                """, unsafe_allow_html=True)

def display_metrics():
    """Display performance metrics."""
    if not st.session_state.performance_metrics:
        st.info("No performance metrics available yet.")
        return
    
    metrics_df = pd.DataFrame(st.session_state.performance_metrics)
    
    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg. Tokens/Second", f"{metrics_df['tokens_per_second'].mean():.2f}")
    with col2:
        st.metric("Total Tokens Processed", f"{metrics_df['total_tokens'].sum():.0f}")
    with col3:
        st.metric("Total Processing Time", f"{metrics_df['elapsed_time_seconds'].sum():.2f}s")
    
    # Display detailed metrics
    st.dataframe(metrics_df)
    
    # Display a line chart of tokens per second over time
    st.line_chart(metrics_df, y='tokens_per_second')

# Main app layout with tabs
st.title("Dr. X's Publications Analyzer")

tabs = st.tabs(["Upload & Process", "Q&A", "Translation", "Summarization", "Performance"])

# Tab 1: Upload & Process
with tabs[0]:
    st.header("Upload and Process Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload Dr. X's publications",
            type=["pdf", "docx", "csv", "xlsx", "xls", "xlsm"],
            accept_multiple_files=True
        )
    
    with col2:
        st.markdown("""
        ### Supported File Types
        - PDF Documents (.pdf)
        - Word Documents (.docx)
        - CSV Files (.csv)
        - Excel Files (.xlsx, .xls, .xlsm)
        """)
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        if st.button("Process Files", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save uploaded file temporarily
                temp_file_path = save_uploaded_file(uploaded_file)
                
                if temp_file_path:
                    try:
                        # Process the document
                        result = st.session_state.analyzer.process_document(temp_file_path)
                        
                        # Add performance metrics
                        metrics = result["performance"]
                        metrics["operation"] = "document_processing"
                        metrics["file"] = uploaded_file.name
                        st.session_state.performance_metrics.append(metrics)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        status_text.text(f"Processed {uploaded_file.name} - Created {result['chunks_created']} chunks")
                        
                        # Clean up temp file
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                            
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        
            status_text.text("All files processed successfully!")
            st.success(f"Processed {len(uploaded_files)} files and added them to the vector database")

# Tab 2: Q&A
with tabs[1]:
    st.header("Ask Questions About Dr. X's Research")
    
    # Display chat history
    display_chat_messages()
    
    # Input for new question
    question = st.text_input("Ask a question about Dr. X's research:")
    col1, col2 = st.columns([1, 5])
    
    with col1:
        n_results = st.number_input("Number of results:", min_value=1, max_value=10, value=5)
    
    with col2:
        if st.button("Ask", type="primary") and question:
            st.session_state.conversation_history.append(("user", question))
            
            with st.spinner("Thinking..."):
                try:
                    # Get answer from RAG system
                    result = st.session_state.analyzer.answer_question(question, n_results)
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append(("assistant", result["answer"]))
                    
                    # Add performance metrics
                    result["performance"]["operation"] = "question_answering"
                    result["performance"]["question"] = question[:30] + "..." if len(question) > 30 else question
                    st.session_state.performance_metrics.append(result["performance"])
                    
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
                    st.session_state.conversation_history.append(("assistant", f"I encountered an error: {str(e)}"))
            
            # Force refresh to display new messages
            st.rerun()
    
    # Reset conversation button
    if st.button("Reset Conversation"):
        st.session_state.analyzer.reset_conversation()
        st.session_state.conversation_history = []
        st.rerun()

# Tab 3: Translation
with tabs[2]:
    st.header("Translate Dr. X's Publications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        translate_file = st.file_uploader(
            "Upload a document to translate",
            type=["pdf", "docx", "csv", "xlsx", "xls", "xlsm"]
        )
        
    with col2:
        source_language = st.selectbox(
            "Source Language:",
            ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Arabic", "Russian", "Auto-detect"]
        )
        
        target_language = st.selectbox(
            "Target Language:",
            ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Arabic", "Russian"]
        )
    
    if translate_file and st.button("Translate", type="primary"):
        with st.spinner("Translating document..."):
            # Save uploaded file temporarily
            temp_file_path = save_uploaded_file(translate_file)
            
            if temp_file_path:
                try:
                    # Translate the document
                    result = st.session_state.analyzer.translate_document(
                        temp_file_path,
                        "Auto-detect" if source_language == "Auto-detect" else source_language,
                        target_language
                    )
                    
                    # Add performance metrics
                    metrics = result["performance"]
                    metrics["operation"] = "translation"
                    metrics["file"] = translate_file.name
                    metrics["source_language"] = source_language
                    metrics["target_language"] = target_language
                    st.session_state.performance_metrics.append(metrics)
                    
                    # Display results
                    st.success(f"Translation completed: {source_language} → {target_language}")
                    
                    for page in result["translated_pages"]:
                        with st.expander(f"Page {page['page_number']}"):
                            st.write(page["content"])
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"Error translating document: {str(e)}")

# Tab 4: Summarization
with tabs[3]:
    st.header("Summarize Dr. X's Publications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        summarize_file = st.file_uploader(
            "Upload a document to summarize",
            type=["pdf", "docx", "csv", "xlsx", "xls", "xlsm"]
        )
        
    with col2:
        summary_ratio = st.slider("Summary Length (% of original)", 10, 50, 30) / 100
    
    if summarize_file and st.button("Summarize", type="primary"):
        with st.spinner("Generating summary..."):
            # Save uploaded file temporarily
            temp_file_path = save_uploaded_file(summarize_file)
            
            if temp_file_path:
                try:
                    # Summarize the document
                    result = st.session_state.analyzer.summarize_document(
                        temp_file_path,
                        summary_ratio
                    )
                    
                    # Add performance metrics
                    metrics = result["performance"]
                    metrics["operation"] = "summarization"
                    metrics["file"] = summarize_file.name
                    st.session_state.performance_metrics.append(metrics)
                    
                    # Display results
                    st.success("Summary generated successfully")
                    
                    st.markdown("### Summary")
                    st.write(result["summary"])
                    
                    st.markdown("### ROUGE Scores")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("ROUGE-1 F1", f"{result['rouge_scores']['rouge-1']['f']:.4f}")
                    with col2:
                        st.metric("ROUGE-2 F1", f"{result['rouge_scores']['rouge-2']['f']:.4f}")
                    with col3:
                        st.metric("ROUGE-L F1", f"{result['rouge_scores']['rouge-l']['f']:.4f}")
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"Error summarizing document: {str(e)}")

# Tab 5: Performance
with tabs[4]:
    st.header("Performance Metrics")
    
    st.markdown("""
    This tab shows performance metrics for various operations performed by the system,
    including tokens per second processed during embedding generation, translation,
    summarization, and RAG processes.
    """)
    
    display_metrics()
    
    if st.button("Clear Metrics"):
        st.session_state.performance_metrics = []
        st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This application analyzes Dr. X's research publications using NLP techniques.
    
    You can:
    - Process multiple document formats
    - Ask questions about the research
    - Translate publications to different languages
    - Generate summaries of the publications
    - Track performance metrics
    """)
    
    st.markdown("---")
    
    st.markdown("## System Information")
    st.markdown(f"**LLM Model:** llama3.2:latest")
    st.markdown(f"**Embedding Model:** nomic-embed-text:latest")
    
    st.markdown("---")
    
    if st.button("Reset Application"):
        # Reset all state
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    import pandas as pd  # Required for metrics display