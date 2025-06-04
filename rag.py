from openai import OpenAI
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize clients
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

es_client = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))
index_name = os.getenv("INDEX_NAME")

def elastic_search(query, course):
    """Search for relevant documents in Elasticsearch"""
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
        result_docs.append(hit['_source'])
    return result_docs

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
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    """Get response from LLM"""
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def rag(query, course):
    """Main RAG function"""
    search_results = elastic_search(query, course)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer, search_results