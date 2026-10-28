import os
import re
import subprocess
import sys
import tempfile
import ollama
from openai import OpenAI

# Explicit client connection to local Ollama server
ollama_client = ollama.Client(host='http://127.0.0.1:11434')

DEFAULT_MAX_ATTEMPTS = 5
EXECUTION_TIMEOUT_SECONDS = 30

SYSTEM_PROMPT = """You are a precise Python coding assistant.

STRICT OUTPUT RULES:
- Respond with ONE Python code block only, using triple backticks (```python ... ```).
- Do NOT wrap the code in any JSON, tool-call format, or function-call syntax.
- Do NOT describe what the code will do or claim you "ran" it — just output the code.
- The code must be complete and runnable as a standalone script (include all imports).
- When generating lists/arrays that must align in length (e.g. for a DataFrame),
  always compute length programmatically: [pool[i % len(pool)] for i in range(n)].
  Never hand-multiply a pattern list and hope the count matches.
- If you are given an error traceback, read it carefully, identify the exact line
  and root cause, and output the complete corrected code (not a diff, not a snippet).
"""


def extract_code(response_text: str) -> str | None:
    """Extracts code block from model response."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def run_code(code: str) -> tuple[bool, str]:
    """Runs code in a temporary subprocess and returns status with output or error."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, (
            f"Execution timed out after {EXECUTION_TIMEOUT_SECONDS} seconds."
        )
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def get_llm_response(provider: str, model_name: str, messages: list, api_key: str = "") -> str:
    """Unified function to fetch response from Local Ollama or Cloud API Providers."""
    if provider == "Local Ollama":
        response = ollama_client.chat(model=model_name, messages=messages)
        return response["message"]["content"]
    
    elif provider == "Groq (Cloud)":
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        response = client.chat.completions.create(model=model_name, messages=messages)
        return response.choices[0].message.content

    elif provider == "OpenAI (Cloud)":
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model=model_name, messages=messages)
        return response.choices[0].message.content

    elif provider == "OpenRouter (Cloud)":
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        response = client.chat.completions.create(model=model_name, messages=messages)
        return response.choices[0].message.content
    
    else:
        raise ValueError(f"Unknown provider: {provider}")