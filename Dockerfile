# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with increased timeout
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application files
COPY app.py .
COPY Models/ Models/
COPY Harcascade/ Harcascade/

# Expose port
EXPOSE 8000

# Run with gunicorn - match Railway's PORT environment variable
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT:-8000}", "--timeout", "180", "--workers", "1", "app:app"]
