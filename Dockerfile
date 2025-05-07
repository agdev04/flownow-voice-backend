FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for pyaudio and other packages
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache utilization
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variable for the API key (to be set during deployment)
ENV GEMINI_API_KEY=""

# Update the code to use environment variable
RUN sed -i 's/api_key="[^"]*"/api_key=os.getenv("GEMINI_API_KEY")/' main.py

# Add import os to main.py
RUN sed -i '1i import os' main.py

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]