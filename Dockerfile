FROM python:3.12-slim

# bash + common CLI tools the agent can use inside the sandbox
RUN apt-get update && apt-get install -y --no-install-recommends bash git tree jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Commands run as an unprivileged user that can only write to the sandbox and data dirs.
RUN useradd --create-home agent && mkdir -p /app/sandbox /app/data/sessions /app/logs \
    && chown -R agent:agent /app/sandbox /app/data /app/logs
USER agent

ENV SHELL_SANDBOX=/app/sandbox SHELL_EXECUTABLE=/bin/bash
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
