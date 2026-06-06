# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (if any are needed by your python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Cloud Run dynamically assigns a PORT environment variable.
# Your main.py already handles this: port=int(os.getenv("PORT", 8000))
# We use uvicorn with workers to handle concurrent USSD/API requests efficiently.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 2