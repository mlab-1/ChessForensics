FROM python:3.12-slim

# Install Stockfish
RUN apt-get update && \
    apt-get install -y --no-install-recommends stockfish && \
    rm -rf /var/lib/apt/lists/*

ENV STOCKFISH_PATH=/usr/games/stockfish

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY chessforensics/ chessforensics/
COPY web/ web/
COPY baselines/ baselines/
COPY server.py ./

# Render uses PORT env var
ENV PORT=10000
EXPOSE 10000

CMD ["python", "-u", "server.py", "--host", "0.0.0.0"]
