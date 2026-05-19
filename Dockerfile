# Use Python 3.9 standard image (not slim - includes needed system libraries)
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with increased timeout
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application files
COPY app.py .
COPY start.sh .
COPY templates/ templates/
COPY Models/ Models/
COPY Harcascade/ Harcascade/

# Make start script executable
RUN chmod +x start.sh

# Expose port
EXPOSE 8000

# Run startup script
CMD ["./start.sh"]
