# LLMScope

LLMScope is an interactive token-level debugger for LLM decoding. It lets you generate text, inspect per-token probabilities, expand alternative continuations, replay generations, and branch from exact token prefixes when the provider supports it.

The app now supports three provider modes:

- `OpenAI`: hosted generation with native token logprobs and alternatives.
- `Ollama`: local REST generation with probabilities unavailable.
- `Hugging Face Local`: direct local `transformers` inference with exact token IDs, probabilities, logprobs, entropy, ranks, and branch continuation from canonical token prefixes.

## Provider Matrix

| Provider | Runs where | Logprobs | Entropy | Branching | Continuation |
| --- | --- | --- | --- | --- | --- |
| OpenAI | Remote | Yes | Yes | Yes | Approximate after cached exact segment ends |
| Ollama | Local via Ollama REST API | No | No | No | No |
| Hugging Face Local | Local via `transformers` + `torch` | Yes | Yes | Yes | Exact from stored token IDs |

## Project Structure

```text
llmscope/
├── backend/
│   ├── app/
│   ├── scripts/
│   ├── tests/
│   ├── requirements.txt
│   └── requirements.local-analysis.txt
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── tests/
│   └── package.json
└── README.md
```

## Prerequisites

- Node.js `20+`
- npm `10+`
- Python `3.10` to `3.14`
- For Hugging Face Local: an NVIDIA GPU with CUDA available to PyTorch

The local-analysis implementation was developed against a Windows + RTX 3060 environment with CUDA-enabled PyTorch already available.

## Backend Setup

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the API:

```powershell
uvicorn app.main:app --reload --port 8000
```

## Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:3000`.

## Hugging Face Local Setup

Hugging Face Local is an optional extension. The base OpenAI/Ollama setup does not require it.

### 1. Install CUDA-enabled PyTorch

Use the current selector at the official PyTorch install page and choose:

- OS: `Windows`
- Package: `Pip`
- Language: `Python`
- Compute Platform: the CUDA version that matches your machine

Official reference: [PyTorch local install guide](https://pytorch.org/get-started/locally/)

PyTorch’s current docs say Windows supports Python `3.10` through `3.14`, and recommend using the command produced by that selector for your CUDA version. Open the selector rather than hard-coding an old wheel URL. Source: [PyTorch docs](https://pytorch.org/get-started/locally/)

### 2. Install local-analysis dependencies

After the correct CUDA-enabled PyTorch wheel is installed:

```powershell
cd backend
pip install -r requirements.local-analysis.txt
```

This installs:

- `torch`
- `transformers`
- `accelerate`
- `safetensors`

### 3. Verify the local runtime

```powershell
cd backend
python scripts/huggingface_local_diagnostics.py
```

The diagnostic script prints:

- PyTorch version
- Transformers version
- CUDA availability
- PyTorch CUDA runtime
- GPU name
- Total and free VRAM
- Selected device
- Selected dtype
- Free disk space

### 4. Load a supported model in the UI

Supported local models:

- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`

The UI will not auto-load or auto-download them. Select `Hugging Face Local`, choose a supported model, and click `Load model`.

The local provider keeps exactly one initialized model/tokenizer pair in memory at a time. Use `Unload model` to release it.

## Hugging Face Local Behavior

- Uses `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`
- Uses the tokenizer chat template with `add_generation_prompt=True`
- Keeps one graph node per actual generated token ID
- Stores raw tokenizer token, decoded contribution, token ID, probability, logprob, rank, entropy, and top alternatives
- Branches from exact canonical token-ID prefixes, not reconstructed UI strings
- Does not silently fall back to CPU
- Prevents overlapping local generations on one GPU

## Exact vs Approximate Continuation

- `OpenAI`: exact while cached provider tokens remain, then approximate continuation starts automatically because the Responses API does not provide true arbitrary assistant-prefill continuation semantics.
- `Hugging Face Local`: exact continuation from stored token IDs as long as the same local model/tokenizer/template/settings remain in use.
- `Ollama`: probabilities and exact continuation are unavailable.

## Cache, Offline Use, and Authentication

Hugging Face’s official environment-variable docs state:

- `HF_HOME` controls where Hugging Face stores local data.
- `HF_HUB_CACHE` controls where repo files are cached.
- By default, they live under `~/.cache/huggingface` and `~/.cache/huggingface/hub`.
- `HF_TOKEN` is optional and overrides the locally stored token if you set it.
- `HF_HUB_OFFLINE=1` forces cached-only usage.

Source: [Hugging Face Hub environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)

For the two supported public Qwen models:

- `HF_TOKEN` is not required.
- No developer-owned secret is baked into LLMScope.
- After the model is cached locally, offline use is possible.

If you choose to override the cache location:

```powershell
$env:HF_HOME="D:\\hf-cache"
```

If you want cached-only mode:

```powershell
$env:HF_HUB_OFFLINE="1"
```

## Optional Manual Smoke Test

After installing local-analysis dependencies:

```powershell
cd backend
python scripts/smoke_huggingface_local.py --model Qwen/Qwen2.5-1.5B-Instruct
```

This will:

1. Load the selected local model.
2. Run one prompt.
3. Print the completion.
4. Print each generated token with raw token, decoded text, token ID, probability, and entropy.

## Ollama Setup

Install Ollama and start the daemon:

```powershell
ollama serve
ollama pull qwen2.5:3b
```

If Ollama is unavailable, LLMScope will show a provider-specific message instead of crashing.

## Environment Variables

Backend:

- `OPENAI_API_KEY`
- `DEFAULT_MODEL`
- `OLLAMA_BASE_URL`
- `HF_TOKEN` optional
- `HF_CONTEXT_LIMIT`
- `HF_DEFAULT_OUTPUT_TOKENS`
- `HF_MAX_OUTPUT_TOKENS`
- `HF_DEFAULT_MODEL`
- `HF_QWEN_1_5B_REVISION`
- `HF_QWEN_3B_REVISION`

Frontend:

- `BACKEND_API_BASE_URL` optional
- `NEXT_PUBLIC_API_BASE_URL` optional

## Local vs Hosted Inference

Local Hugging Face analysis uses the GPU on the machine running the backend.

- If you run LLMScope on your own PC, it uses your local GPU.
- If you deploy LLMScope on a server, it uses the server owner’s GPU.

The browser alone does not run `transformers` inference on the visitor’s GPU.

## API Surface

Main backend endpoints:

- `GET /health`
- `GET /models`
- `GET /models/refresh`
- `POST /generate`
- `POST /expand-node`
- `POST /continue-node`
- `GET /providers/hugging-face-local`
- `POST /providers/hugging-face-local/load`
- `POST /providers/hugging-face-local/unload`
- `GET /providers/hugging-face-local/diagnostics`

## Testing

Backend:

```powershell
python -m unittest discover -s backend/tests -p "test_*.py"
```

Frontend:

```powershell
cd frontend
npm run typecheck
npm run lint
npm exec --package=tsx tsx --test tests/provider-model-selection.test.ts tests/token-graph-alternative-expansion.test.ts
```

## Troubleshooting

### CUDA unavailable

- Confirm `nvidia-smi` works.
- Confirm the installed PyTorch build is CUDA-enabled.
- Run `python scripts/huggingface_local_diagnostics.py`.

### CUDA OOM

- Reduce max output tokens.
- Explore shorter branches.
- Unload the current model and load `Qwen/Qwen2.5-1.5B-Instruct`.
- Close other GPU-heavy apps.

### Missing Hugging Face dependencies

Install the optional local-analysis requirements:

```powershell
pip install -r backend/requirements.local-analysis.txt
```

### Delete cached model files

Remove the cache directory configured by `HF_HOME` / `HF_HUB_CACHE`, or the default Hugging Face cache under your user profile.

## Licensing Notes

- LLMScope code and model weights are separate things.
- Supported Hugging Face models keep their own upstream licenses.
- Downloading a model locally does not change the model’s license terms.
