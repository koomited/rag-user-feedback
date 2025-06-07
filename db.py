import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        port=os.getenv("POSTGRES_PORT")
    )

def init_db():
    """Initialize database tables by dropping them if they exist, then recreating."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            print("[INFO] Dropping existing tables (if any)...")
            cur.execute("DROP TABLE IF EXISTS feedback")
            cur.execute("DROP TABLE IF EXISTS conversations")

            print("[INFO] Creating 'conversations' table...")
            cur.execute("""
                CREATE TABLE conversations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    course VARCHAR(100) NOT NULL,
                    model_used VARCHAR(100),
                    response_time FLOAT,
                    relevance VARCHAR(20),
                    relevance_explanation TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    gemini_cost FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            print("[INFO] Creating 'feedback' table...")
            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id UUID REFERENCES conversations(id),
                    feedback INTEGER NOT NULL CHECK (feedback IN (-1, 1)),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            print("[INFO] Database initialized successfully.")
    finally:
        conn.close()

def save_conversation(question, answer, course, model_used=None, response_time=None, 
                     relevance=None, relevance_explanation=None, prompt_tokens=None, 
                     completion_tokens=None, gemini_cost=None):
    """Save a conversation to the database"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            conversation_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO conversations 
                (id, question, answer, course, model_used, response_time, 
                 relevance, relevance_explanation, prompt_tokens, completion_tokens, gemini_cost)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (conversation_id, question, answer, course, model_used, response_time,
                  relevance, relevance_explanation, prompt_tokens, completion_tokens, gemini_cost))
            
            result = cur.fetchone()
            conn.commit()
            return result[0] if result else conversation_id
    finally:
        conn.close()

def save_feedback(conversation_id, feedback):
    """Save user feedback for a conversation"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feedback (conversation_id, feedback)
                VALUES (%s, %s)
            """, (conversation_id, feedback))
            conn.commit()
    finally:
        conn.close()

def get_conversation_by_id(conversation_id):
    """Get a conversation by ID"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM conversations WHERE id = %s
            """, (conversation_id,))
            return cur.fetchone()
    finally:
        conn.close()

def get_feedback_stats():
    """Get feedback statistics"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_feedback,
                    COUNT(CASE WHEN feedback = 1 THEN 1 END) as positive_feedback,
                    COUNT(CASE WHEN feedback = -1 THEN 1 END) as negative_feedback
                FROM feedback
            """)
            return cur.fetchone()
    finally:
        conn.close()

def get_last_conversations(limit=5, relevance_filter=None):
    """Get the last N conversations, optionally filtered by relevance"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT id, question, answer, course, model_used, response_time,
                       relevance, relevance_explanation, prompt_tokens,
                       completion_tokens, gemini_cost, timestamp
                FROM conversations
            """
            params = []
            if relevance_filter:
                query += " WHERE relevance = %s"
                params.append(relevance_filter)
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            return cur.fetchall()
    finally:
        conn.close()