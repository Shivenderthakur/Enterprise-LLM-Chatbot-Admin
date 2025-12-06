sudo apt-get update && apt-get install -y lshw
curl -fsSL https://ollama.com/install.sh | sh
pip install streamlit requests pypdf langchain langchain-community sentence-transformers pymongo fastapi python-dotenv
pip uninstall -y fitz      # remove the wrong one if present
pip install --upgrade pymupdf
pip install fastapi uvicorn

