import os
import time
from datetime import datetime, timedelta, timezone
import random
from faker import Faker
from dotenv import load_dotenv
import logging
from db import save_conversation, save_feedback
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker for synthetic data
fake = Faker()

# Load environment variables
load_dotenv()
logger.info("Loading environment variables...")

# Configuration
COURSES = [
    "data-engineering-zoomcamp",
    "machine-learning-zoomcamp",
    "mlops-zoomcamp"
]
MODELS = [
    "ollama (phi3)",
    "gemini-1.5-flash (gemini)",
    "gemini-1.5-flash-8b (gemini)",
    "gemini-2.0-flash (gemini)"
]
RELEVANCE_VALUES = ["RELEVANT", "PARTIALLY_RELEVANT", "NON_RELEVANT", "UNKNOWN"]
FEEDBACK_VALUES = [1, -1, 0]  # 1: Thumbs up, -1: Thumbs down, 0: Neutral

def generate_question(course):
    """Generate a course-specific question"""
    if course == "data-engineering-zoomcamp":
        templates = [
            "How do I set up {} in {}?",
            "What is the difference between {} and {}?",
            "Can I use {} for {} in the course?",
            "How to configure {} for {}?"
        ]
        techs = ["Apache Kafka", "PostgreSQL", "Docker", "Airflow", "Spark"]
        return random.choice(templates).format(random.choice(techs), random.choice(techs))
    elif course == "machine-learning-zoomcamp":
        templates = [
            "How to implement {} in {}?",
            "What is the purpose of {} in machine learning?",
            "Can I use {} with {}?",
            "How to tune {} for better performance?"
        ]
        techs = ["scikit-learn", "TensorFlow", "XGBoost", "neural networks", "SVM"]
        return random.choice(templates).format(random.choice(techs), random.choice(techs))
    else:  # mlops-zoomcamp
        templates = [
            "How to deploy {} using {}?",
            "What is the role of {} in MLOps?",
            "How to monitor {} with {}?",
            "Can I integrate {} with {}?"
        ]
        techs = ["Kubernetes", "MLflow", "Prometheus", "Grafana", "Seldon"]
        return random.choice(templates).format(random.choice(techs), random.choice(techs))

def generate_answer(question):
    """Generate a synthetic answer based on the question"""
    return fake.text(max_nb_chars=200).replace("\n", " ") + f" For more details, refer to the {random.choice(['course FAQ', 'module notes', 'official documentation'])}."

def generate_feedback(conversation_id):
    """Generate feedback for a conversation"""
    if random.random() < 0.5:  # 50% chance of feedback
        feedback = random.choice(FEEDBACK_VALUES)
        try:
            feedback_id = save_feedback(
                conversation_id=conversation_id,
                feedback=feedback
            )
            logger.info(f"Inserted feedback ID {feedback_id} for conversation ID {conversation_id} with value {feedback}")
            return feedback_id
        except Exception as e:
            logger.error(f"Failed to insert feedback for conversation ID {conversation_id}: {e}")
    return None

def generate_historical_data(num_records=50):
    """Generate historical data up to 6 hours ago"""
    logger.info(f"Generating {num_records} historical records...")
    now = datetime.now(timezone.utc)
    six_hours_ago = now - timedelta(hours=6)
    
    for _ in range(num_records):
        # Random timestamp between now and 6 hours ago
        timestamp = six_hours_ago + timedelta(seconds=random.randint(0, 6*3600))
        course = random.choice(COURSES)
        question = generate_question(course)
        answer = generate_answer(question)
        model_used = random.choice(MODELS)
        response_time = round(random.uniform(0.5, 5.0), 2)
        relevance = random.choice(RELEVANCE_VALUES)
        relevance_explanation = fake.sentence(nb_words=10).replace(".", "")
        prompt_tokens = random.randint(100, 1000)
        completion_tokens = random.randint(50, 500)
        gemini_cost = round(random.uniform(0.0, 0.01), 4) if "gemini" in model_used else 0.0
        
        try:
            conversation_id = save_conversation(
                question=question,
                answer=answer,
                course=course,
                model_used=model_used,
                response_time=response_time,
                relevance=relevance,
                relevance_explanation=relevance_explanation,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                gemini_cost=gemini_cost
            )
            logger.info(f"Inserted historical record ID {conversation_id} with timestamp {timestamp}")
            
            # Generate feedback for this conversation
            generate_feedback(conversation_id)
        except Exception as e:
            logger.error(f"Failed to insert historical record: {e}")

def generate_live_data():
    """Generate live data every second"""
    logger.info("Starting live data generation...")
    try:
        while True:
            now = datetime.now(timezone.utc)
            course = random.choice(COURSES)
            question = generate_question(course)
            answer = generate_answer(question)
            model_used = random.choice(MODELS)
            response_time = round(random.uniform(0.5, 5.0), 2)
            relevance = random.choice(RELEVANCE_VALUES)
            relevance_explanation = fake.sentence(nb_words=10).replace(".", "")
            prompt_tokens = random.randint(100, 1000)
            completion_tokens = random.randint(50, 500)
            gemini_cost = round(random.uniform(0.0, 0.01), 4) if "gemini" in model_used else 0.0
            
            try:
                conversation_id = save_conversation(
                    question=question,
                    answer=answer,
                    course=course,
                    model_used=model_used,
                    response_time=response_time,
                    relevance=relevance,
                    relevance_explanation=relevance_explanation,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    gemini_cost=gemini_cost
                )
                logger.info(f"Inserted live record ID {conversation_id} with timestamp {now}")
                
                # Generate feedback for this conversation
                generate_feedback(conversation_id)
            except Exception as e:
                logger.error(f"Failed to insert live record: {e}")
            
            time.sleep(1)  # Wait 1 second before next insertion
    except KeyboardInterrupt:
        logger.info("Stopped live data generation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for course assistant")
    parser.add_argument("--historical", action="store_true", help="Generate only historical data")
    args = parser.parse_args()

    # Generate historical data
    generate_historical_data(num_records=50)
    
    # Generate live data only if --historical is not specified
    if not args.historical:
        generate_live_data()