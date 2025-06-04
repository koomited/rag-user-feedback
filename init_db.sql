-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    course VARCHAR(100) NOT NULL,
    model_used VARCHAR(100),
    response_time FLOAT,
    relevance_score FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    feedback INTEGER NOT NULL CHECK (feedback IN (-1, 1)),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_conversations_course ON conversations(course);
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_conversation_id ON feedback(conversation_id);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_value ON feedback(feedback);

-- Add some constraints for data integrity
ALTER TABLE conversations 
ADD CONSTRAINT chk_response_time_positive 
CHECK (response_time IS NULL OR response_time >= 0);

ALTER TABLE conversations 
ADD CONSTRAINT chk_relevance_score_range 
CHECK (relevance_score IS NULL OR (relevance_score >= 0 AND relevance_score <= 1));

-- Create a view for conversation analytics
CREATE OR REPLACE VIEW conversation_analytics AS
SELECT 
    c.course,
    COUNT(*) as total_conversations,
    AVG(c.response_time) as avg_response_time,
    AVG(c.relevance_score) as avg_relevance_score,
    COUNT(f.feedback) as feedback_count,
    COUNT(CASE WHEN f.feedback = 1 THEN 1 END) as positive_feedback,
    COUNT(CASE WHEN f.feedback = -1 THEN 1 END) as negative_feedback,
    CASE 
        WHEN COUNT(f.feedback) > 0 
        THEN ROUND(COUNT(CASE WHEN f.feedback = 1 THEN 1 END)::NUMERIC / COUNT(f.feedback) * 100, 2)
        ELSE NULL 
    END as positive_feedback_percentage
FROM conversations c
LEFT JOIN feedback f ON c.id = f.conversation_id
GROUP BY c.course;

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO course_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO course_user;