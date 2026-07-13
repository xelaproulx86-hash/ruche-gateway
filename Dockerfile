# 🐝 ruche-gateway — image CPU pour endpoint RunPod load-balancing
FROM python:3.11-slim

WORKDIR /app

# git requis par la greffe hivebase (dormante sans GITHUB_TOKEN)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py builtin_tools.py ./

# RunPod LB injecte PORT; défaut local 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
