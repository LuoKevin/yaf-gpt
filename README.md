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

<!--
Markdown Cheat Sheet
====================

Headings
--------
# H1
## H2
### H3

Emphasis
--------
*italic* or _italic_
**bold** or __bold__
~~strikethrough~~

Lists
-----
- Unordered item
  - Nested item
1. Ordered item
2. Second item

Code
----
Inline: `code`
Block:
```python
print("Hello, world!")
```

Links & Images
--------------
[link text](https://example.com)
![alt text](path/to/img.png)

Quotes & Tables
---------------
> Blockquote

| Col A | Col B |
| ----- | ----- |
| Val 1 | Val 2 |

Horizontal Rule
---------------
---
-->
