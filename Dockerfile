FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN python -m pip install --no-cache-dir -e ".[dev]"
CMD ["python", "scripts/run_all.py", "--mode", "fixture"]
