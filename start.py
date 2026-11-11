#!/usr/bin/env python3
"""
Startup script - Launches both FastAPI backend and Streamlit frontend
"""

import os
import sys
import subprocess
import time
import platform

# Get the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Python executable path
if platform.system() == "Windows":
    PYTHON_EXE = os.path.join(SCRIPT_DIR, "venv", "Scripts", "python.exe")
else:
    PYTHON_EXE = os.path.join(SCRIPT_DIR, "venv", "bin", "python")

# Check if virtual environment exists
if not os.path.exists(PYTHON_EXE):
    print("Virtual environment not found!")
    print(f"Expected at: {PYTHON_EXE}")
    sys.exit(1)

print("RAG Chatbot Startup")
print("=" * 60)

# Start FastAPI backend
print("\nStarting FastAPI backend on port 8001...")
fastapi_cmd = [
    PYTHON_EXE,
    "-m", "uvicorn",
    "rag_chatbot.main:app",
    "--host", "0.0.0.0",
    "--port", "8001"
]

try:
    backend_process = subprocess.Popen(fastapi_cmd)
    print("✓ FastAPI backend started (PID: {})".format(backend_process.pid))
except Exception as e:
    print(f"Failed to start FastAPI backend: {e}")
    sys.exit(1)

# Wait for backend to start
print("waiting for backend to initialize...")
time.sleep(3)

# Start Streamlit frontend
print("\nStarting Streamlit frontend on port 8501...")
streamlit_cmd = [
    PYTHON_EXE,
    "-m", "streamlit", "run",
    "streamlit_app.py",
    "--server.port", "8501",
    "--server.address", "localhost"
]

try:
    frontend_process = subprocess.Popen(streamlit_cmd)
    print("✓ Streamlit frontend started (PID: {})".format(frontend_process.pid))
except Exception as e:
    print(f"Failed to start Streamlit frontend: {e}")
    backend_process.terminate()
    sys.exit(1)

print("\n" + "=" * 60)
print("RAG Chatbot is running!")
print("\nOpen in your browser:")
print("Streamlit UI: http://localhost:8501")
print("API Docs:     http://localhost:8001/docs")
print("\nTips:")
print("   - Use Streamlit UI for easy interaction")
print("   - Use API docs for advanced testing")
print("   - Check sidebar for diagnostics and timing tests")
print("   - Press Ctrl+C to stop both services")
print("=" * 60 + "\n")

# Keep the processes running
try:
    backend_process.wait()
    frontend_process.wait()
except KeyboardInterrupt:
    print("\n\nShutting down...")
    backend_process.terminate()
    frontend_process.terminate()
    
    # Give them time to shut down gracefully
    time.sleep(1)
    
    # Force kill if needed
    try:
        backend_process.kill()
        frontend_process.kill()
    except:
        pass
    
    print("Services stopped")
    sys.exit(0)
