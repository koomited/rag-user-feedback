import streamlit as st
import time
import os
from dotenv import load_dotenv
from rag import rag
from db import save_conversation, save_feedback, get_last_conversations, get_feedback_stats
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

COURSES = {
    "machine-learning-zoomcamp": "Machine Learning Zoomcamp",
    "data-engineering-zoomcamp": "Data Engineering Zoomcamp", 
    "mlops-zoomcamp": "MLOps Zoomcamp"
}

MODEL_PROVIDERS = {
    "ollama": "Ollama (Local)",
    "gemini-1.5-flash": "Gemini 1.5 Flash (Cloud)",
    "gemini-1.5-flash-8b": "Gemini 1.5 Flash 8b (Cloud)",
    "gemini-2.0-flash": "Gemini 2.0 Flash (Cloud)"
}

SEARCH_TYPES = {
    "text": "Text Search",
    "vector": "Vector Search"
}

# Initialize session state
if 'model_provider' not in st.session_state:
    st.session_state.model_provider = "ollama"
if 'search_type' not in st.session_state:
    st.session_state.search_type = "text"
if 'relevance_filter' not in st.session_state:
    st.session_state.relevance_filter = None

st.title("🎓 Course Assistant - RAG Q&A")
st.markdown("Ask questions about your course materials and get instant answers!")

# Configuration section
st.markdown("### ⚙️ Configuration")
col1, col2 = st.columns(2)

with col1:
    selected_provider = st.selectbox(
        "Model Provider:",
        options=list(MODEL_PROVIDERS.keys()),
        format_func=lambda x: MODEL_PROVIDERS[x],
        index=0,
        help="Choose between local Ollama or cloud-based Gemini"
    )
    st.session_state.model_provider = selected_provider

with col2:
    selected_search = st.selectbox(
        "Search Type:",
        options=list(SEARCH_TYPES.keys()),
        format_func=lambda x: SEARCH_TYPES[x],
        index=0,
        help="Text search uses keywords, Vector search uses semantic similarity"
    )
    st.session_state.search_type = selected_search

selected_course = st.selectbox(
    "Select your course:",
    options=list(COURSES.keys()),
    format_func=lambda x: COURSES[x],
    index=1
)

with st.expander("🔧 Current Settings", expanded=False):
    st.write(f"**Model Provider:** {MODEL_PROVIDERS[st.session_state.model_provider]}")
    st.write(f"**Search Type:** {SEARCH_TYPES[st.session_state.search_type]}")
    st.write(f"**Course:** {COURSES[selected_course]}")
    
    if st.session_state.model_provider == "ollama":
        st.write(f"**Model:** {os.getenv('LLM_MODEL', 'phi3')}")
        st.write(f"**Base URL:** {os.getenv('OPENAI_BASE_URL', 'http://localhost:11434/v1')}")
    else:
        st.write(f"**Model:** {st.session_state.model_provider}")

st.markdown("---")

# Input box
user_input = st.text_input(
    "Enter your question:", 
    placeholder="e.g., How do I set up Apache Kafka?",
    help="Ask any question about your selected course materials"
)

# Button to trigger RAG
if st.button("🔍 Ask", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            try:
                logger.info(f"Calling rag with query: {user_input}, course: {selected_course}, model: {st.session_state.model_provider}, search_type: {st.session_state.search_type}")
                result = rag(
                    query=user_input, 
                    course=selected_course,
                    model_provider=st.session_state.model_provider,
                    search_type=st.session_state.search_type
                )
                logger.info("RAG function returned successfully")
                
                conversation_id = save_conversation(
                    question=user_input,
                    answer=result["answer"],
                    course=selected_course,
                    model_used=result["model_used"],
                    response_time=result["response_time"],
                    relevance=result["relevance"],
                    relevance_explanation=result["relevance_explanation"],
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    gemini_cost=result["gemini_cost"]
                )
                logger.info(f"Conversation saved with ID: {conversation_id}")
                
                st.session_state.conversation_id = conversation_id
                st.session_state.current_answer = result["answer"]
                st.session_state.search_results = result["search_results"]
                st.session_state.response_time = result["response_time"]
                st.session_state.relevance = result["relevance"]
                st.session_state.relevance_explanation = result["relevance_explanation"]
                
            except Exception as e:
                logger.error(f"RAG call failed: {e}")
                st.error(f"❌ Error: {str(e)}")
                st.session_state.conversation_id = None
                result = None

# Display results
if hasattr(st.session_state, 'current_answer') and st.session_state.current_answer:
    st.success(f"✅ Done! (Response time: {st.session_state.response_time:.2f}s)")
    
    st.markdown("### 💡 Answer:")
    st.markdown(st.session_state.current_answer)
    
    st.markdown(f"**Relevance:** {st.session_state.relevance}")
    st.markdown(f"**Relevance Explanation:** {st.session_state.relevance_explanation}")
    
    if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
        with st.expander(f"📚 Source Documents ({len(st.session_state.search_results)} found)", expanded=False):
            for i, doc in enumerate(st.session_state.search_results, 1):
                st.markdown(f"**📄 Document {i}**")
                if 'section' in doc:
                    st.markdown(f"**Section:** {doc['section']}")
                if 'question' in doc:
                    st.markdown(f"**Question:** {doc['question']}")
                if 'text' in doc:
                    st.markdown(f"**Content:** {doc['text'][:200]}{'...' if len(doc['text']) > 200 else ''}")
                if '_score' in doc:
                    st.markdown(f"**Relevance Score:** {doc['_score']:.3f}")
                if i < len(st.session_state.search_results):
                    st.markdown("---")
    
    st.markdown("### 📝 Was this answer helpful?")
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("👍", key="positive"):
            if hasattr(st.session_state, 'conversation_id') and st.session_state.conversation_id:
                save_feedback(st.session_state.conversation_id, 1)
                st.success("Thanks for your positive feedback!")
            else:
                st.error("No conversation to rate")
    
    with col2:
        if st.button("👎", key="negative"):
            if hasattr(st.session_state, 'conversation_id') and st.session_state.conversation_id:
                save_feedback(st.session_state.conversation_id, -1)
                st.success("Thanks for your feedback! We'll work to improve.")
            else:
                st.error("No conversation to rate")

# Last 5 conversations
st.markdown("---")
st.markdown("### 📜 Recent Conversations")
st.selectbox(
    "Filter by relevance:",
    options=[None, "RELEVANT", "PARTIALLY_RELEVANT", "NON_RELEVANT", "UNKNOWN"],
    format_func=lambda x: "All" if x is None else x,
    key="relevance_filter"
)

try:
    conversations = get_last_conversations(limit=5, relevance_filter=st.session_state.relevance_filter)
    logger.info(f"Retrieved {len(conversations)} recent conversations")
    if conversations:
        for conv in conversations:
            with st.expander(f"Q: {conv['question'][:50]}{'...' if len(conv['question']) > 50 else ''} ({conv['timestamp']})"):
                st.markdown(f"**Course:** {conv['course']}")
                st.markdown(f"**Model:** {conv['model_used']}")
                st.markdown(f"**Response Time:** {conv['response_time']:.2f}s")
                st.markdown(f"**Relevance:** {conv['relevance']}")
                st.markdown(f"**Relevance Explanation:** {conv['relevance_explanation']}")
                st.markdown(f"**Prompt Tokens:** {conv['prompt_tokens']}")
                st.markdown(f"**Completion Tokens:** {conv['completion_tokens']}")
                st.markdown(f"**Gemini Cost:** ${conv['gemini_cost']:.4f}" if conv['gemini_cost'] else "**Gemini Cost:** N/A")
                st.markdown(f"**Question:** {conv['question']}")
                st.markdown(f"**Answer:** {conv['answer'][:200]}{'...' if len(conv['answer']) > 200 else ''}")
    else:
        st.write("No conversations found.")
except Exception as e:
    logger.error(f"Failed to retrieve conversations: {e}")
    st.error(f"❌ Error retrieving conversations: {str(e)}")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Current Selection")
    st.markdown(f"**Course:** {COURSES[selected_course]}")
    st.markdown(f"**Provider:** {MODEL_PROVIDERS[st.session_state.model_provider]}")
    st.markdown(f"**Search:** {SEARCH_TYPES[st.session_state.search_type]}")
    
    st.markdown("---")
    st.markdown("### 📚 Available Courses")
    for course_key, course_name in COURSES.items():
        if course_key == selected_course:
            st.markdown(f"**🎯 {course_name}**")
        else:
            st.markdown(f"• {course_name}")
    
    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. **Configure** your preferences
    2. **Select** your course
    3. **Ask** your question  
    4. **Get** AI-powered answers
    5. **Rate** the response
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Technology Stack")
    st.markdown("""
    **Search:**
    - 🔍 Elasticsearch (Text/Vector)
    
    **AI Models:**
    - 🏠 Ollama (Local)
    - ☁️ Gemini (Cloud)
    
    **Storage:**
    - 🗄️ PostgreSQL
    
    **Monitoring:**
    - 📊 Grafana
    """)
    
    st.markdown("---")
    st.markdown("### 🔋 System Status")
    
    ollama_status = "🟢 Available" if os.getenv("OPENAI_BASE_URL") else "🔴 Not configured"
    st.markdown(f"**Ollama:** {ollama_status}")
    
    gemini_status = "🟢 Available" if os.getenv("GEMINI_API_KEY") else "🔴 Not configured"
    st.markdown(f"**Gemini:** {gemini_status}")
    
    es_status = "🟢 Available" if os.getenv("ELASTICSEARCH_URL") else "🔴 Not configured"
    st.markdown(f"**Elasticsearch:** {es_status}")
    
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Made with ❤️ using Streamlit</div>", 
    unsafe_allow_html=True
)