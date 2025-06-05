import os
import requests
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from db import init_db

# Load environment variables
def load_environment():
    print("[INFO] Loading environment variables...")
    load_dotenv()

# Download documents from remote URL
def download_documents() -> list:
    print("[INFO] Downloading documents from GitHub...")
    base_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"
    relative_url = '05-best-practices/documents-with-ids.json'
    docs_url = f"{base_url}/{relative_url}?raw=1"

    response = requests.get(docs_url)
    response.raise_for_status()

    print(f"[INFO] Successfully downloaded {len(response.json())} documents.")
    return response.json()

# Create Elasticsearch client
def create_elasticsearch_client() -> Elasticsearch:
    print("[INFO] Connecting to Elasticsearch...")
    
    es = Elasticsearch(
        "http://elasticsearch:9200",
    )
    return es

# Define index settings and create index
def setup_index(es_client: Elasticsearch, index_name: str):
    print(index_name)
    print(f"[INFO] Deleting existing index '{index_name}' (if any)...")
    es_client.indices.delete(index=index_name, ignore_unavailable=True)

    print(f"[INFO] Creating index '{index_name}'...")
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"},
                "question_text_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
            }
        }
    }

    es_client.indices.create(index=index_name, body=index_settings, request_timeout=90)
    print("[INFO] Index created successfully.")

# Encode and index documents
def index_documents(es_client: Elasticsearch, model: SentenceTransformer, documents: list, index_name: str):
    print("[INFO] Indexing documents...")
    for doc in tqdm(documents):
        question = doc["question"]
        text = doc["text"]
        combined_text = question + " " + text
        doc["question_text_vector"] = model.encode(combined_text)
        es_client.index(index=index_name, document=doc)
    print("[INFO] All documents indexed successfully.")

# Main execution
def main():
    load_environment()

    documents = download_documents()
    es_client = create_elasticsearch_client()
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    index_name = "course-questions"
    setup_index(es_client, index_name)
    index_documents(es_client, model, documents, index_name)

    print("[INFO] Initializing PostgreSQL database...")
    init_db()
    print("[INFO] Setup complete.")

if __name__ == "__main__":
    main()
