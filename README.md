# LLMScope

LLMScope is an interactive developer tool for inspecting how large language models generate text. The long-term product direction includes token-by-token playback, probability inspection, alternative predictions, latency and cost analysis, and side-by-side comparison across cloud and local models. This initial scaffold focuses on clean architecture, typed API contracts, and a production-ready dashboard shell without implementing live generation yet.

## Architecture

### Frontend

- **Framework:** Next.js 15, React 19, TypeScript, Tailwind CSS
- **Purpose:** Render the inspection workspace, prompt controls, token trace surface, and statistics panels
- **Current state:** Uses typed mock data and local UI state so the visual structure can mature before backend streaming is wired

### Backend

- **Framework:** FastAPI with Pydantic models
- **Purpose:** Provide a stable HTTP contract for health checks and generation responses
- **Current state:** Exposes `GET /health` and `POST /generate`, with `/generate` returning mock traced output shaped for future real model execution

### Containerization

- `docker/backend.Dockerfile`: FastAPI runtime image
- `docker/frontend.Dockerfile`: Next.js standalone production image
- `docker-compose.yml`: Local multi-service orchestration for frontend and backend

## Project Structure

```text
llmscope/
├── frontend/
│   ├── app/
│   ├── components/
│   ├── hooks/
│   ├── lib/
│   ├── public/
│   ├── types/
│   ├── .env.example
│   └── package.json
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   ├── .env.example
│   └── requirements.txt
├── docker/
├── docs/
├── .gitignore
├── docker-compose.yml
└── README.md
```

## Prerequisites

- Node.js `20.19+` recommended for the current Next.js 15 toolchain
- npm `10+`
- Python `3.12+`

## Setup

### 1. Configure environment files

```powershell
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.example frontend/.env
```

### 2. Start the backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`.

### 3. Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`.

### 4. Optional: run with Docker Compose

After creating `backend/.env` and `frontend/.env`:

```powershell
docker compose up --build
```

## Available Endpoints

- `GET /health`: Service health and runtime metadata
- `POST /generate`: Mock generation payload shaped for future token-level tracing

Example request:

```json
{
  "prompt": "Explain the difference between beam search and nucleus sampling.",
  "model": "gpt-4.1-mini",
  "max_tokens": 256,
  "temperature": 0.7
}
```

## Development Roadmap

1. Replace the mock generation service with an OpenAI-backed implementation.
2. Stream token events from the backend to the frontend.
3. Visualize alternative candidates, token timings, and confidence bands.
4. Add pricing models and detailed latency instrumentation.
5. Introduce a provider abstraction for local Hugging Face execution.
6. Support side-by-side multi-model comparison and exportable traces.

## Future Features

- Token-by-token playback with live updates
- Top-k / top-p alternative token inspection
- Cost and latency breakdown by request and token
- Parallel provider comparisons
- Saved sessions, shareable traces, and reproducible experiments
- Local inference adapters for Hugging Face and other runtime backends

## Notes

- The current UI intentionally stops at scaffold state. The panels are interactive enough to validate layout and data contracts, but they do not call the backend yet.
- The backend service contract is already structured around future provider expansion.
- Additional architectural notes live in [docs/architecture.md](docs/architecture.md).
