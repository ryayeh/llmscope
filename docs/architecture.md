# LLMScope Architecture

LLMScope starts as a split frontend/backend application so token inspection UX and model execution logic can evolve independently.

## Frontend

- `app/`: Next.js App Router entrypoints and global styling.
- `components/`: Reusable layout, panel, and UI components for the inspection dashboard.
- `hooks/`: Client-side state hooks used to stage prompt inputs and placeholder selection state.
- `lib/`: Shared helpers and mock data that let the interface render meaningful states before the live API is connected.
- `types/`: UI-facing TypeScript contracts for tokens, metrics, navigation, and roadmap items.

## Backend

- `app/api/`: Route registration and request handlers.
- `app/core/`: Configuration and environment loading.
- `app/models/`: Shared backend domain enums and provider identifiers.
- `app/schemas/`: Pydantic request and response contracts.
- `app/services/`: Generation service boundary where OpenAI and future Hugging Face clients will be introduced.

## Runtime Shape

1. The Next.js client captures prompt inputs and request options.
2. The frontend calls the FastAPI backend over HTTP.
3. The backend normalizes request settings, invokes a provider service, and returns traced generation data.
4. The frontend renders token-level and run-level analytics from that structured payload.

The current scaffold stops at step 3 with a mock response so the contract is stable before live generation logic is introduced.
