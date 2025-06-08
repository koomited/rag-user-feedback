from openai import OpenAI
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer
import json
import time
from google import genai
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize clients
def get_ollama_client():
    """Initialize Ollama client"""
    try:
        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "ollama")
        )
        logger.info("Ollama client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Ollama client: {e}")
        raise

def get_gemini_client():
    """Initialize Gemini client"""
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        logger.info("Gemini client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        raise

# Initialize Elasticsearch
try:
    es_client = Elasticsearch(
        os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200"),
        headers={
            "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
            "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
        }
    )
    logger.info("Elasticsearch client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Elasticsearch client: {e}")
    raise

index_name = os.getenv("INDEX_NAME", "course-questions")

def elastic_search_text(query, course):
    """Search for relevant documents in Elasticsearch using text search"""
    try:
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
        logger.info(f"Text search for query '{query}' returned {len(result_docs)} results")
        return result_docs
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        raise

def elastic_search_vector(vector, course, field="question_text_vector"):
    """Search for relevant documents using KNN vector search"""
    try:
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
        logger.info(f"Vector search for course '{course}' returned {len(result_docs)} results")
        return result_docs
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise

def get_query_embedding(query):
    """Generate embedding for the query using SentenceTransformer"""
    try:
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        embedding = model.encode(query)
        logger.info(f"Generated embedding for query: {query}")
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return None

def build_prompt(query, search_results):
    """Build the prompt for the LLM"""
    try:
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
        logger.info("Prompt built successfully")
        return prompt
    except Exception as e:
        logger.error(f"Failed to build prompt: {e}")
        raise

def llm_ollama(prompt, model=None):
    """Get response from Ollama"""
    try:
        client = get_ollama_client()
        model = model or os.getenv("LLM_MODEL", "phi3")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        logger.info(f"Ollama response received for model {model}")
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        raise

def estimate_tokens(text):
    """Estimate token count based on word count (approximation for Gemini)"""
    try:
        words = len(text.split())
        token_count = int(words / 0.75)
        logger.info(f"Estimated {token_count} tokens for text length {len(text)}")
        return token_count
    except Exception as e:
        logger.error(f"Token estimation failed: {e}")
        return 0

def llm_gemini(prompt, model="gemini-1.5-flash"):
    """Get response from Gemini"""
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=model, 
            contents=prompt
        )
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(response.text)
        gemini_cost = (prompt_tokens * 0.00035 + completion_tokens * 0.00105) / 1000
        logger.info(f"Gemini response received for model {model}")
        return response.text, prompt_tokens, completion_tokens, gemini_cost
    except Exception as e:
        logger.error(f"Gemini request failed: {e}")
        raise

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

{{
   "Relevance": "NON_RELEVANT" | "PARTIALLY_RELEVANT" | "RELEVANT",
   "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()
    
    try:
        prompt = prompt_template.format(question=question, answer=answer)
        client = get_gemini_client()
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        # Clean the response: remove code blocks, extra newlines, and whitespace
        cleaned_response = re.sub(r'```json\n|```|\n\s*\n', '\n', response.text).strip()
        cleaned_response = re.sub(r'^\s*|\s*$', '', cleaned_response)
        # Ensure response starts and ends with curly braces
        if not cleaned_response.startswith('{'):
            cleaned_response = '{' + cleaned_response
        if not cleaned_response.endswith('}'):
            cleaned_response = cleaned_response + '}'
        logger.debug(f"Cleaned response: {cleaned_response}")
        json_eval = json.loads(cleaned_response)
        logger.info(f"Relevance evaluated: {json_eval['Relevance']}")
        return json_eval["Relevance"], json_eval["Explanation"]
    except Exception as e:
        logger.error(f"Failed to parse relevance evaluation: {e}")
        logger.error(f"Raw response: {getattr(response, 'text', 'No response available')}")
        return "UNKNOWN", f"Failed to evaluate relevance due to parsing error: {str(e)}"

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
    try:
        logger.info(f"Starting RAG for query: {query}, course: {course}, model: {model_provider}, search_type: {search_type}")
        start_time = time.time()
        
        # Perform search based on type
        if search_type == "vector":
            vector = get_query_embedding(query)
            if vector is None:
                logger.warning("Failed to generate embedding, falling back to text search")
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
        
        result = {
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
        logger.info("RAG completed successfully")
        return result
    except Exception as e:
        logger.error(f"RAG function failed: {e}")
        raise

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