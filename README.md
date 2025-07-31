# AI Course assistant

<p align="center">
  <img src="images/banner.png">
</p>

## Project overview
This project was completed in the framework of an LLM zoomcamp where the organisers have FAQ google docs documents on three different courses: llm zoomcamp, data engineering zoomcamp and machine learning zoomcamp. THe objective here is to make life easier to student by creation a chatbot that can automatically repond to questions that already exist in the FAQ documents. The students save time and do not need to go through the whole FAQ document to find a single answer.


The dataset used in this project contains information about the three different courses, including:

- **section:** The name of the section the question belongs to.
- **question:** A previous question answerd in the FAQ.
- **course:** The course on which the question is asked.
- **id:** A generated Id that uniquely represent each question and the related information.
- **ext:** The answer to the question.


You can find the data in [`data/documents-with-ids.json](data/documents-with-ids.json).

## Technologies used
* ElasticSearch hybrid search (vector and text) 
* GEMINI as LLM
* Streamlit for user interface
* PostGRES as database
* Grafana for monitoring
* Docker compose for services running


## Running locally 

First, installing the dependencies 

```bash
pip install -r requirements.txt
```
Then start the services

```bash
docker compose up -d
```
And finally open the streamlit application:
It's accessible at [localhost:8501](http://localhost:8501)
```

## Code

- [`app.py`](app.py) - the Streamlit user interface
- [`rag.py`](rag.py) - the main RAG logic for building the retrieving the data and building the prompt
- [`generate_data.py`](generate_data.py) - generate fake data for testing
- [`db.py`](db.py) - the logic for logging the requests and responses to postgres
- [`prep.py`](prep.py) - the script for initializing the database

## Monitoring

We use Grafana for monitoring the application. 

It's accessible at [localhost:3000](http://localhost:3000):

- Login: "admin"
- Password: "admin"

The monitoring dashboard contains several panels:

1. **Last 5 Conversations (Table):** Displays a table showing the five most recent conversations, including details such as the question, answer, relevance, and timestamp. This panel helps monitor recent interactions with users.
2. **+1/-1 (Pie Chart):** A pie chart that visualizes the feedback from users, showing the count of positive (thumbs up) and negative (thumbs down) feedback received. This panel helps track user satisfaction.
3. **Relevancy (Gauge):** A gauge chart representing the relevance of the responses provided during conversations. The chart categorizes relevance and indicates thresholds using different colors to highlight varying levels of response quality.
4. **GEMINI Cost (Time Series):** A time series line chart depicting the cost associated with GEMINI usage over time. This panel helps monitor and analyze the expenditure linked to the AI model's usage.
5. **Tokens (Time Series):** Another time series chart that tracks the number of tokens used in conversations over time. This helps to understand the usage patterns and the volume of data processed.
6. **Model Used (Bar Chart):** A bar chart displaying the count of conversations based on the different models used. This panel provides insights into which AI models are most frequently used.
7. **Response Time (Time Series):** A time series chart showing the response time of conversations over time. This panel is useful for identifying performance issues and ensuring the system's responsiveness.