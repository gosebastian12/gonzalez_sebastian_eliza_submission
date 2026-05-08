# Gonzalez Sebastian ELIZA submission

Python **3.10+** (see `.python-version`). Install dependencies with [uv](https://docs.astral.sh/uv/) from the repo root: `uv sync`.

## ChromaDB (local vector database)

This project uses [Chroma](https://www.trychroma.com/) as an open-source vector store for RAG: you **ingest** embedded document chunks and **query** them for retrieval. The server runs in Docker using the official image **`chromadb/chroma:1.5.3`**, with data persisted in a named volume so collections survive container restarts.

| Item | Value |
| --- | --- |
| Compose file | `src/backend/database/docker-compose.yaml` |
| Server config | `src/backend/database/chroma.docker.yaml` (listen address, persistence path) |
| HTTP API | `http://127.0.0.1:8001` (host from your machine: `localhost`) |
| Persistence | Docker volume `chroma_data` → `/data` inside the container |

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Compose v2 (`docker compose`).

### Start the database

From the repository root:

```bash
docker compose -f src/backend/database/docker-compose.yaml up -d
```

Or from `src/backend/database`:

```bash
cd src/backend/database
docker compose up -d
```

Wait until the service is healthy (the compose file defines a heartbeat healthcheck). Check status:

```bash
docker compose -f src/backend/database/docker-compose.yaml ps
```

### Verify it is running

**HTTP (API v2 heartbeat):**

```bash
curl -sS http://127.0.0.1:8001/api/v2/heartbeat
```

**Python (`chromadb` HTTP client):**

```python
import chromadb

client = chromadb.HttpClient(host="localhost", port=8001)
client.heartbeat()
```

Use the same host and port when configuring **LangChain** `Chroma` / vector store helpers that talk to a remote Chroma server (point them at `http://localhost:8001` per library docs).

### Stop and data retention

Stop containers without deleting stored vectors:

```bash
docker compose -f src/backend/database/docker-compose.yaml down
```

To remove the named volume and **wipe** local Chroma data:

```bash
docker compose -f src/backend/database/docker-compose.yaml down -v
```

`chroma.docker.yaml` sets `allow_reset: true` so collections can be reset during development; set it to `false` if you want to disallow reset operations from clients.
