# ⚡ AutoCoder Local — Self-Correcting AI Coding Agent

**Comprehensive Technical Documentation, Post-Mortem & Architecture Guide**

---

## 1. Project Overview & Context

**AutoCoder Local** is a local, self-correcting AI coding agent designed to run seamlessly on your workstation using **Ollama** and a **Streamlit** user interface. It automatically generates Python scripts based on user prompt specifications, executes them inside an isolated subprocess environment, captures execution tracebacks upon failure, and recursively feeds errors back to the LLM to self-correct until a fully working script is verified.

### Core Problems Solved:
1. **Open WebUI Code Interpreter Failure:** Standard tool-calling JSON schemas cause recurrent parsing crashes on 7B local models. AutoCoder Local bypasses JSON schemas using direct Python regular expressions (`extract_code`) to parse clean triple-backtick Markdown code blocks.
2. **Infinite Error Loops:** Standard execution environments stop on error. AutoCoder Local implements a **Configurable Attempt Feedback Loop** (default: 5 attempts) that feeds exact tracebacks back to the LLM for automated correction.
3. **Local & Cloud Hybrid Handoff:** System supports both local Ollama models (`127.0.0.1:11434`) and cloud API providers (Groq, OpenAI, OpenRouter) with secure password-style API key inputs.
4. **Modern Claude/Gemini-Inspired UI:** Features a sleek dark theme (`#0d1117`), zero top-padding gap, live server/API key status indicators, and a real-time **Execution Summary & Attempts Log** tracking specific errors per attempt.

---

## 2. Technical Stack & Architecture

* **Frontend UI:** Streamlit with custom CSS (Claude/Gemini dark theme, glassmorphic cards).
* **AI Orchestration & API Router:** `ollama-python` (Local binding) & `openai` SDK (Cloud API compatibility layer).
* **Runtime Execution Engine:** Standard Python Subprocess (`subprocess.run`) with temporary file creation and a 30-second execution timeout.
* **Tested Hardware Profile:** Windows PC with NVIDIA GeForce RTX 3050 GPU, 16GB RAM.

```
AutoCoder_Local/
├── requirements.txt      # Dependency Manifest
├── agent_core.py         # Execution Runtime & Multi-LLM Gateway
└── app.py                # Streamlit Web UI & Execution Log Tracker
```

---

## 3. Complete Codebase Implementation

### File 1: `requirements.txt`
```text
ollama>=0.3.0
pandas>=2.2.0
streamlit>=1.30.0
openai>=1.0.0
requests>=2.31.0
```

---

### File 2: `agent_core.py`
```python
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
```

---

### File 3: `app.py`
```python
import time
import streamlit as st
from agent_core import EXECUTION_TIMEOUT_SECONDS, DEFAULT_MAX_ATTEMPTS, SYSTEM_PROMPT, extract_code, run_code, get_llm_response, ollama_client

# Streamlit Page Setup
st.set_page_config(
    page_title="AutoCoder Local | Next-Gen AI Coding Environment",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS with top padding reduction & clean cards
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color-scheme: dark;
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    .stApp, [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
    }
    
    header {
        background-color: transparent !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .sub-title {
        color: #8b949e;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>div>textarea {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #f0f6fc !important;
    }
    
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        border: none !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 12px rgba(46,160,67,0.3) !important;
    }
    
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(46,160,67,0.4) !important;
    }

    .summary-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        height: 100%;
    }
    
    .summary-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #58a6ff;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-header">
    <div class="main-title">⚡ AutoCoder Hybrid</div>
    <div class="sub-title">Next-Generation Self-Correcting AI Code Studio with Hybrid Provider Support</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Engine & API Settings")
    st.markdown("---")
    
    provider = st.selectbox(
        "Select AI Provider",
        options=["Local Ollama", "Groq (Cloud)", "OpenAI (Cloud)", "OpenRouter (Cloud)"],
        help="Choose between running locally via Ollama or using cloud API providers."
    )
    
    api_key = ""
    if provider != "Local Ollama":
        api_key = st.text_input(
            f"Enter {provider} API Key",
            type="password",
            placeholder="sk-...",
            help="Your API key is securely handled in session and never stored permanently."
        )

    if provider == "Local Ollama":
        local_models = []
        try:
            models_response = ollama_client.list()
            if hasattr(models_response, 'models'):
                local_models = [m.model for m in models_response.models]
            elif isinstance(models_response, dict) and 'models' in models_response:
                local_models = [m.get('name') or m.get('model') for m in models_response['models']]
        except Exception:
            local_models = []

        if not local_models:
            local_models = ["qwen2.5-coder:7b", "llama3.1:8b", "mistral:7b"]

        model_selection = st.selectbox("Select Local Model", options=local_models + ["Custom Model Name..."])
        if model_selection == "Custom Model Name...":
            model_name = st.text_input("Model Identifier", value="qwen2.5-coder:7b")
        else:
            model_name = model_selection
            
    elif provider == "Groq (Cloud)":
        model_name = st.selectbox("Select Groq Model", options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    elif provider == "OpenAI (Cloud)":
        model_name = st.selectbox("Select OpenAI Model", options=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    elif provider == "OpenRouter (Cloud)":
        model_name = st.text_input("OpenRouter Model ID", value="anthropic/claude-3.5-sonnet")

    max_attempts = st.number_input(
        "Max Correction Retry Attempts", 
        min_value=1, 
        max_value=10, 
        value=DEFAULT_MAX_ATTEMPTS
    )
    
    custom_filename = st.text_input("Export Code Filename", value="agent_output.py")
    
    st.markdown("---")
    st.markdown("**Connection Status:**")
    if provider == "Local Ollama":
        try:
            ollama_client.list()
            st.success("🟢 Local Ollama Connected")
        except Exception:
            st.error("🔴 Ollama Server Offline")
    else:
        if api_key.strip():
            st.success(f"🟢 {provider} Key Configured")
        else:
            st.warning(f"🟡 Waiting for {provider} Key")

# Session state initialization
if "attempt_logs" not in st.session_state:
    st.session_state.attempt_logs = []
if "success_attempt" not in st.session_state:
    st.session_state.success_attempt = None

# Workspace Section
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("#### 🎯 Define Your Coding Task")
    
    preset_options = [
        "Custom Problem (Type your own prompt below)",
        "Generate mock data for 50 employees with EmployeeID, Department (HR, Finance, IT, Marketing), and Salary. Print DataFrame shape.",
        "Read a sample CSV file, calculate mean & standard deviation of numerical columns, and output summary.",
        "Create an in-memory SQLite database, create a table, insert 10 sample user records, and query active users.",
        "Write a Python script to fetch sample JSON data from a dummy API, parse it, and display formatted keys.",
    ]

    selected_option = st.selectbox("Quick Templates:", preset_options)

    if selected_option == "Custom Problem (Type your own prompt below)":
        task_input = st.text_area(
            "Prompt Specification:",
            placeholder="e.g. Build a script to generate synthetic sales data, apply standard scaling, and save to CSV...",
            height=130,
        )
    else:
        task_input = selected_option

with col_right:
    st.markdown("#### 📋 Execution Summary & Attempts Log")
    
    logs_html = '<div class="summary-card">'
    logs_html += f'<div class="summary-title"><span>📊 Status & Attempt Tracker</span></div>'
    
    if st.session_state.success_attempt:
        logs_html += f'<p style="color: #2ea043; font-weight:600; margin-bottom:8px;">✅ Success on Attempt #{st.session_state.success_attempt}!</p>'
    elif st.session_state.attempt_logs:
        logs_html += f'<p style="color: #f85149; font-weight:600; margin-bottom:8px;">❌ Running / Failed after {len(st.session_state.attempt_logs)} attempts</p>'
    else:
        logs_html += f'<p style="color: #8b949e; font-size: 0.9rem;">Awaiting execution. Click run below to start tracking attempts and errors.</p>'
    
    if st.session_state.attempt_logs:
        logs_html += '<div style="max-height: 120px; overflow-y: auto; font-size: 0.8rem; font-family: monospace; background: #0d1117; padding: 8px; border-radius: 6px; border: 1px solid #30363d;">'
        for log in st.session_state.attempt_logs:
            logs_html += f'<b style="color: #58a6ff;">Attempt {log["attempt"]}:</b> <span style="color: #ff7b72;">{log["error_summary"]}</span><br>'
        logs_html += '</div>'
        
    logs_html += '</div>'
    st.markdown(logs_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Run Execution Trigger
if st.button("🚀 Execute AutoCoder Agent", type="primary", use_container_width=True):
    if not task_input.strip():
        st.warning("⚠️ Please provide a valid task specification or prompt before running!")
    elif provider != "Local Ollama" and not api_key.strip():
        st.error(f"⚠️ Please enter a valid API key for {provider} in the sidebar.")
    else:
        st.session_state.attempt_logs = []
        st.session_state.success_attempt = None
        
        st.markdown(f"**Status:** Processing prompt using **`{model_name}`** via **{provider}**...")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_input},
        ]

        success_flag = False
        start_time = time.time()

        for attempt in range(1, max_attempts + 1):
            with st.expander(
                f"🔄 Attempt {attempt} of {max_attempts}",
                expanded=(attempt == 1 or not success_flag),
            ):
                st.write(f"🤖 *Querying {provider} engine for standalone code...*")

                try:
                    reply_text = get_llm_response(provider, model_name, messages, api_key)
                except Exception as e:
                    st.error(f"❌ API / Connection Error: {e}")
                    break

                code = extract_code(reply_text)

                if not code:
                    error_msg = "Model response missing valid python code block."
                    st.warning(f"⚠️ {error_msg}")
                    st.session_state.attempt_logs.append({
                        "attempt": attempt,
                        "error_summary": error_msg
                    })
                    messages.append({"role": "assistant", "content": reply_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": "You must respond with a single ```python code block. Try again.",
                        }
                    )
                    continue

                st.markdown("##### 📜 Generated Python Code")
                st.code(code, language="python")

                st.write("⚡ *Executing code inside isolated subprocess...*")
                success, output = run_code(code)

                if success:
                    elapsed_time = round(time.time() - start_time, 2)
                    st.session_state.success_attempt = attempt
                    st.success(f"✅ Executed Successfully on Attempt {attempt} in {elapsed_time}s!")
                    st.markdown("##### 🖨️ Execution Output (stdout)")
                    st.code(output if output else "Script executed cleanly with no stdout output.")

                    final_filename = custom_filename.strip()
                    if not final_filename.endswith(".py"):
                        final_filename += ".py"

                    st.download_button(
                        label=f"📥 Download Verified Code ({final_filename})",
                        data=code,
                        file_name=final_filename,
                        mime="text/x-python",
                        use_container_width=True
                    )
                    success_flag = True
                    break
                else:
                    err_lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
                    short_err = err_lines[-1] if err_lines else "Unknown execution error"
                    if len(short_err) > 60:
                        short_err = short_err[:57] + "..."

                    st.session_state.attempt_logs.append({
                        "attempt": attempt,
                        "error_summary": short_err
                    })

                    st.error("❌ Execution Error Traceback:")
                    st.code(output, language="bash")

                    if attempt == max_attempts:
                        st.error(
                            f"⛔ Maximum retry limit reached ({max_attempts}). Auto-correction failed."
                        )
                    else:
                        st.info(f"💡 Attempt {attempt} failed. Feeding error back to model for self-correction...")
                        messages.append({"role": "assistant", "content": reply_text})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Running that code produced this error:\n\n"
                                    f"{output}\n\n"
                                    "Fix the code and output the complete corrected version as a single python code block."
                                ),
                            }
                        )
```

---

## 4. Execution Guide

1. Clone or copy `requirements.txt`, `agent_core.py`, and `app.py` into your working folder (`AutoCoder_Local`).
2. Activate your Python virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start your local Ollama server (`ollama serve` or standard service).
5. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```