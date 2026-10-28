"""
llm.py - talks to the local Ollama server and streams the answer back.

Ollama's /api/chat endpoint with "stream": true does NOT send one big JSON
reply. It sends many small JSON objects, one per line ("newline-delimited
JSON"), as the model generates tokens:

  {"message": {"role": "assistant", "content": "REST"}, "done": false}
  {"message": {"role": "assistant", "content": " is"},  "done": false}
  ...
  {"message": {"role": "assistant", "content": ""}, "done": true, "total_duration": ...}

stream_answer() is a *generator*: it yields each "content" piece as soon as
its line arrives, so the UI can show the answer word by word instead of
waiting for the whole thing.
"""

import json

import requests

import config

CHAT_URL = f"{config.OLLAMA_URL}/api/chat"

NOT_RUNNING_MSG = (
    f"Can't reach Ollama at {config.OLLAMA_URL}. Is it running?\n"
    "Start it with:  ollama serve   (or open the Ollama app)"
)
NOT_PULLED_MSG = (
    f"The model '{config.OLLAMA_MODEL}' isn't downloaded yet.\n"
    f"Run:  ollama pull {config.OLLAMA_MODEL}"
)


class LLMError(Exception):
    """A friendly, user-facing error about the LLM (Ollama down, model missing, ...)."""


def _post(payload):
    """POST to /api/chat with streaming on, turning low-level errors into LLMError."""
    try:
        response = requests.post(
            CHAT_URL,
            json=payload,
            stream=True,  # don't wait for the whole body - let us read it line by line
            # (connect timeout, read timeout). The read timeout is the max gap
            # between two pieces of data, not the total answer time.
            timeout=(5, config.OLLAMA_TIMEOUT),
        )
    except requests.ConnectionError as e:
        raise LLMError(NOT_RUNNING_MSG) from e
    except requests.Timeout as e:
        raise LLMError("Ollama took too long to respond.") from e

    if response.status_code == 404:  # Ollama returns 404 when the model isn't pulled
        response.close()
        raise LLMError(NOT_PULLED_MSG)
    if response.status_code != 200:
        message = response.text
        response.close()
        raise LLMError(f"Ollama returned HTTP {response.status_code}: {message}")
    return response


def stream_answer(question):
    """Yield the answer to `question` piece by piece as Ollama generates it."""
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "stream": True,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }

    # 'with' makes sure the HTTP connection is closed even if the caller stops early.
    with _post(payload) as response:
        try:
            for line in response.iter_lines():
                if not line:  # skip blank keep-alive lines
                    continue
                data = json.loads(line)
                if "error" in data:  # Ollama can report an error mid-stream
                    raise LLMError(f"Ollama error: {data['error']}")
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break
        except requests.RequestException as e:  # connection dropped mid-answer
            raise LLMError(f"Lost connection to Ollama: {e}") from e


def warm_up():
    """Load the model into memory ahead of time, so the first real question is fast.

    A chat request with an empty message list makes Ollama load the model and
    return immediately without generating anything.
    """
    payload = {"model": config.OLLAMA_MODEL, "messages": [], "keep_alive": config.OLLAMA_KEEP_ALIVE}
    with _post(payload) as response:
        response.content  # read the (tiny) reply so the request completes
