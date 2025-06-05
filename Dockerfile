# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (e.g., curl to check service availability)
RUN apt-get update && apt-get install -y curl && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment settings
ENV PYTHONUNBUFFERED=1
ENV ELASTICSEARCH_URL=http://elasticsearch:9200

# Expose Streamlit port
EXPOSE 8501

# CMD: Wait for Elasticsearch → run prep.py → run Streamlit app
CMD bash -c "\
  echo 'Waiting for Elasticsearch at $ELASTICSEARCH_URL...'; \
  until curl -s $ELASTICSEARCH_URL >/dev/null; do \
    echo 'Elasticsearch not available yet. Retrying in 2 seconds...'; \
    sleep 2; \
  done; \
  echo 'Elasticsearch is up. Running prep.py...'; \
  python prep.py; \
  echo 'Starting Streamlit app...'; \
  streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none"
