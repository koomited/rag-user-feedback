-- Average Response Time
SELECT AVG(response_time) as avg_response_time
FROM conversations
WHERE timestamp >= NOW() - INTERVAL '1 hour';

-- Fraction of Relevant Answers
SELECT 
    relevance,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
FROM conversations
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY relevance;

-- Thumbs Up/Down Stats
SELECT 
    f.feedback,
    COUNT(*) as count
FROM feedback f
JOIN conversations c ON f.conversation_id = c.id
WHERE c.timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY f.feedback;

-- Last 5 Conversations
SELECT 
    question,
    answer,
    course,
    model_used,
    response_time,
    relevance,
    timestamp
FROM conversations
ORDER BY timestamp DESC
LIMIT 5;

-- Token Usage
SELECT 
    model_used,
    AVG(prompt_tokens) as avg_prompt_tokens,
    AVG(completion_tokens) as avg_completion_tokens,
    AVG(gemini_cost) as avg_gemini_cost
FROM conversations
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY model_used;