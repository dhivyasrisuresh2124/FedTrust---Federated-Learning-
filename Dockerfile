FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# CPU-only PyTorch keeps the image small
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install Flask requests numpy scipy scikit-learn gunicorn
COPY . .
ENV PYTHONPATH=/app
EXPOSE 5000 5001
# Default: a campus node. docker-compose overrides the command for the coordinator.
CMD ["python", "-m", "campus.app", "--host", "0.0.0.0", "--port", "5001"]
