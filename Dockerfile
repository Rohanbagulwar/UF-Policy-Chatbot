FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY req.txt .
RUN pip install -r req.txt

# Copy application code
COPY . .
COPY static/ static/

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
