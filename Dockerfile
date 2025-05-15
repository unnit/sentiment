# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy app code
COPY . .

# Expose port (same for dev and prod)
EXPOSE 5003

# Environment variable to switch between dev/prod
ENV FLASK_ENV=production

# Default command (will be overridden by docker-compose)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5003", "app:app"]
