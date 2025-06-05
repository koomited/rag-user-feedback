import streamlit as st
import time
import os
from dotenv import load_dotenv
from rag import rag
from db import init_db, save_conversation, save_feedback

# Load environment variables
load_dotenv()

# Initialize database on startup
@st.cache_resource
def initialize_database():
    init_db()
    return True

initialize_database()

# Course options
COURSES = {
    "machine-learning-zoomcamp": "Machine Learning Zoomcamp",
    "data-engineering-zoomcamp": "Data Engineering Zoomcamp", 
    "mlops-zoomcamp": "MLOps Zoomcamp"
}

# Streamlit UI
st.title("🎓 Course Assistant - RAG Q&A")
st.markdown("Ask questions about your course materials and get instant answers!")

# Course selection
selected_course = st.selectbox(
    "Select your course:",
    options=list(COURSES.keys()),
    format_func=lambda x: COURSES[x],
    index=1  # Default to data-engineering-zoomcamp
)

# Input box
user_input = st.text_input("Enter your question:", placeholder="e.g., How do I set up Apache Kafka?")

# Button to trigger RAG
if st.button("🔍 Ask", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        # Show loading spinner while processing
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            try:
                answer, search_results = rag(user_input, selected_course)
                response_time = time.time() - start_time
                
                # Save conversation to database
                conversation_id = save_conversation(
                    question=user_input,
                    answer=answer,
                    course=selected_course,
                    model_used=os.getenv("LLM_MODEL"),
                    response_time=response_time
                )
                
                # Store conversation ID in session state
                st.session_state.conversation_id = conversation_id
                st.session_state.current_answer = answer
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.conversation_id = None
                answer = None

# Display results if we have an answer
if hasattr(st.session_state, 'current_answer') and st.session_state.current_answer:
    st.success("✅ Done!")
    
    # Display the answer
    st.markdown("### 💡 Answer:")
    st.markdown(st.session_state.current_answer)
    
    # Feedback buttons
    st.markdown("### 📝 Was this answer helpful?")
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("👍 +1", key="positive"):
            if hasattr(st.session_state, 'conversation_id') and st.session_state.conversation_id:
                save_feedback(st.session_state.conversation_id, 1)
                st.success("Thanks for your positive feedback!")
            else:
                st.error("No conversation to rate")
    
    with col2:
        if st.button("👎 -1", key="negative"):
            if hasattr(st.session_state, 'conversation_id') and st.session_state.conversation_id:
                save_feedback(st.session_state.conversation_id, -1)
                st.success("Thanks for your feedback! We'll work to improve.")
            else:
                st.error("No conversation to rate")

# Sidebar with information
with st.sidebar:
    st.markdown("### 📚 Available Courses")
    for course_key, course_name in COURSES.items():
        if course_key == selected_course:
            st.markdown(f"**🎯 {course_name}**")
        else:
            st.markdown(f"• {course_name}")
    
    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. **Select** your course
    2. **Ask** your question  
    3. **Get** AI-powered answers
    4. **Rate** the response
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Powered by")
    st.markdown("""
    - **Elasticsearch** for search
    - **Ollama/Phi3** for AI responses
    - **PostgreSQL** for data storage
    """)