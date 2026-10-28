"""
Test llm.py on its own: stream an answer in the terminal and measure latency.

  python -m scripts.test_llm                          # asks "What is REST API?"
  python -m scripts.test_llm "Explain SQL vs NoSQL"   # ask your own question
"""

import sys
import time

import llm

question = sys.argv[1] if len(sys.argv) > 1 else "What is REST API?"
print(f"Q: {question}\nA: ", end="", flush=True)

start = time.perf_counter()
first_token = None
try:
    for piece in llm.stream_answer(question):
        if first_token is None:
            first_token = time.perf_counter() - start
        print(piece, end="", flush=True)  # flush so each piece appears immediately
except llm.LLMError as e:
    print(f"\nERROR: {e}")
    raise SystemExit(1)
total = time.perf_counter() - start

print(f"\n\nTime to first token : {first_token * 1000:.0f} ms")
print(f"Total time          : {total * 1000:.0f} ms")
