from openai import OpenAI
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer
import json
import time
from google import genai

load_dotenv()

# Initialize clients
def get_ollama_client():
    """Initialize Ollama client"""
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "ollama")
    )

def get_gemini_client():
    """Initialize Gemini client"""
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Elasticsearch
es_client = Elasticsearch(
    os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200"),
    headers={
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
    }
)
index_name = os.getenv("INDEX_NAME", "course-questions")

def elastic_search_text(query, course):
    """Search for relevant documents in Elasticsearch using text search"""
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": course
                    }
                }
            }
        }
    }
    
    response = es_client.search(index=index_name, body=search_query)
    result_docs = []
    for hit in response['hits']['hits']:
        doc = hit['_source']
        doc['_score'] = hit['_score']
        result_docs.append(doc)
    return result_docs

def elastic_search_vector(vector, course, field="question_text_vector"):
    """Search for relevant documents using KNN vector search"""
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {
            "term": {
                "course": course
            }
        }
    }
    
    response = es_client.search(
        index=index_name,
        knn=knn,
        source=["id", "question", "text", "section", "course"]
    )
    
    result_docs = []
    for hit in response['hits']['hits']:
        doc = hit['_source']
        doc['_score'] = hit['_score']
        result_docs.append(doc)
    return result_docs

def get_query_embedding(query):
    """Generate embedding for the query using SentenceTransformer"""
    try:
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        embedding = model.encode(query)
        return embedding.tolist()
    except Exception as e:
        print(f"Failed to generate embedding: {e}")
        return None

def build_prompt(query, search_results):
    """Build the prompt for the LLM"""
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()
    
    context = ""
    for doc in search_results:
        context += f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm_ollama(prompt, model=None):
    """Get response from Ollama"""
    client = get_ollama_client()
    model = model or os.getenv("LLM_MODEL", "phi3")
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

def estimate_tokens(text):
    """Estimate token count based on word count (approximation for Gemini)"""
    # Average tokens per word is ~0.75 for English text in modern LLMs
    words = len(text.split())
    return int(words / 0.75)

def llm_gemini(prompt, model="gemini-1.5-flash"):
    """Get response from Gemini"""
    client = get_gemini_client()
    response = client.models.generate_content(
        model=model, 
        contents=prompt
    )
    # Estimate token counts for Gemini
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(response.text)
    # Mock cost calculation (example: $0.00035 per 1K input tokens, $0.00105 per 1K output tokens)
    gemini_cost = (prompt_tokens * 0.00035 + completion_tokens * 0.00105) / 1000
    return response.text, prompt_tokens, completion_tokens, gemini_cost

def evaluate_relevance(question, answer):
    """Evaluate answer relevance using Gemini-1.5-flash"""
    prompt_template = """
Your are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

Your task is to analyse the relevance of the generated answer to the given question.
Based on the relevance of the answer, you will classify it as :
"NON_RELEVANT", "PARTIALLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:
Question: {question}
Generated Answer: {answer}

Please analyse the content and context of the generated answer in relation to the question and provide your evaluation in a parsable JSON format without using code blocks:

{
   "Relevance": "NON_RELEVANT" | "PARTIALLY_RELEVANT" | "RELEVANT",
   "Explanation": "[Provide a brief explanation for your evaluation]"
}
""".strip()
    
    prompt = prompt_template.format(question=question, answer=answer)
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        json_eval = json.loads(response.text)
        return json_eval["Relevance"], json_eval["Explanation"]
    except Exception as e:
        print(f"Failed to parse relevance evaluation: {e}")
        return "UNKNOWN", "Failed to evaluate relevance due to parsing error"

def rag(query, course, model_provider="ollama", search_type="text"):
    """
    Main RAG function with enhanced capabilities
    
    Args:
        query (str): The user's question
        course (str): The course identifier
        model_provider (str): "ollama" or a Gemini model (e.g., "gemini-1.5-flash")
        search_type (str): "text" or "vector"
    
    Returns:
        dict: Contains answer, search_results, response_time, model_used, prompt_tokens,
              completion_tokens, gemini_cost, relevance, relevance_explanation
    """
    start_time = time.time()
    
    # Perform search based on type
    if search_type == "vector":
        vector = get_query_embedding(query)
        if vector is None:
            print("Failed to generate embedding, falling back to text search")
            search_results = elastic_search_text(query, course)
        else:
            search_results = elastic_search_vector(vector, course, field="question_text_vector")
    else:
        search_results = elastic_search_text(query, course)
    
    # Build prompt
    prompt = build_prompt(query, search_results)
    
    # Get LLM response based on provider
    if model_provider == "ollama":
        answer, prompt_tokens, completion_tokens = llm_ollama(prompt)
        gemini_cost = 0.0  # No cost for local Ollama
        model_used = os.getenv("LLM_MODEL", "phi3")
    else:
        answer, prompt_tokens, completion_tokens, gemini_cost = llm_gemini(prompt, model=model_provider)
        model_used = model_provider
    
    # Evaluate relevance
    relevance, relevance_explanation = evaluate_relevance(query, answer)
    
    response_time = time.time() - start_time
    
    return {
        "answer": answer,
        "search_results": search_results,
        "response_time": response_time,
        "model_used": f"{model_used} ({'ollama' if model_provider == 'ollama' else 'gemini'})",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "gemini_cost": gemini_cost,
        "relevance": relevance,
        "relevance_explanation": relevance_explanation
    }

# Backward compatibility
def elastic_search(query, course):
    """Original function for backward compatibility"""
    return elastic_search_text(query, course)

def elastic_search_knn_text(field, vector, course):
    """Original function signature for backward compatibility"""
    return elastic_search_vector(vector, course, field)

def llm(prompt, model="gemini-1.5-flash"):
    """Original function for backward compatibility"""
    text, _, _, _ = llm_gemini(prompt, model)
    return text