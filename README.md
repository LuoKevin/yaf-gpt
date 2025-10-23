# yaf-gpt

## Young Adult Fellowship Generative Pre-trained Transformer

## A chatbot to assist with Bible Study sessions.

## Quick Start

Prepare a virtual environment, install dependencies, then launch the FastAPI app.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn yaf_gpt.interface.api:app --reload
```

## Usage

Once the server is running locally:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

You should receive a JSON payload containing a stubbed assistant reply.

## LLM Configuration

The chat service routes requests through an `LLMClient` abstraction. By default it uses an offline stub so the API works without external credentials. To talk to OpenAI:

1. Install the `openai` package (e.g., `pip install openai`).
2. Export `OPENAI_API_KEY`.
3. Instantiate `ChatService` with `OpenAIChatClient` (see `yaf_gpt/model/llm_client.py` for details).