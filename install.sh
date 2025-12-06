#!/usr/bin/env bash
set -e

### CONFIGURATION #########################################################

# Project paths (adjust if needed)
PROJECT_DIR="/opt/llm-admin"
PYTHON_BIN="python3"          # or python3.11 etc.
VENV_DIR="$PROJECT_DIR/venv"

# Service ports
STREAMLIT_PORT=8501
API_PORT=8000                 # VectorApp / FastAPI
OLLAMA_PORT=11434

# Mongo / Vector settings (edit for your environment)
export MONGODBURI="YOUR_MONGODB_ATLAS_URI"
export KBDBNAME="vectordb"
export KBCOLLECTION="pdfembeddings"
export VECTORFIELD="embedding"
export VECTORINDEXNAME="vectorindex"
export VECTORDIM=768
export GPUMEMORYLIMITBYTES=$((14*1024*1024*1024))   # ~14 GB
export EMBEDMODEL="BAAI/bge-base-en-v1.5"
export VECTORDB="$MONGODBURI"   # used in VectorApp.py

# Admin credentials
export APP_ADMIN_USER="admin"
export APP_ADMIN_PASS="admin"

# Backend base URL used by AdminApp to reach FastAPI
export ADMINAPIBASE="http://localhost:${API_PORT}"

###########################################################################

echo "[+] Creating project directory (if missing)..."
sudo mkdir -p "$PROJECT_DIR"
sudo chown "$(whoami)":"$(whoami)" "$PROJECT_DIR"

echo "[+] Copying repo files into $PROJECT_DIR (assumes you ran this from repo root)..."
cp -r . "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "[+] Updating system packages..."
sudo apt-get update
sudo apt-get install -y curl python3 python3-venv python3-pip lshw

echo "[+] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "[+] Creating Python virtual environment..."
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[+] Upgrading pip..."
pip install --upgrade pip

echo "[+] Installing Python dependencies..."
pip install streamlit requests pypdf langchain langchain-community \
           sentence-transformers pymongo fastapi python-dotenv uvicorn

# Fix PyMuPDF / fitz naming issue (as in original install.sh)[file:5]
pip uninstall -y fitz || true
pip install --upgrade pymupdf

echo "[+] Writing .env file..."
cat > "$PROJECT_DIR/.env" <<EOF
MONGODBURI=${MONGODBURI}
KBDBNAME=${KBDBNAME}
KBCOLLECTION=${KBCOLLECTION}
VECTORFIELD=${VECTORFIELD}
VECTORINDEXNAME=${VECTORINDEXNAME}
VECTORDIM=${VECTORDIM}
GPUMEMORYLIMITBYTES=${GPUMEMORYLIMITBYTES}
EMBEDMODEL=${EMBEDMODEL}
APPADMINUSER=${APP_ADMIN_USER}
APPADMINPASS=${APP_ADMIN_PASS}
ADMINAPIBASE=${ADMINAPIBASE}
VECTORDB=${VECTORDB}
STREAMLITPORT=${STREAMLIT_PORT}
APIPORT=${API_PORT}
EOF

echo "[+] Pulling at least one Ollama model (edit as needed)..."
# Example: llama3.2; change to your preferred model
ollama pull llama3.2 || true

echo "[+] Starting Ollama service (if not already running)..."
if ! pgrep -x "ollama" >/dev/null; then
  ollama serve > "$PROJECT_DIR/log_ollama.out" 2>&1 &
  OLLAMA_PID=$!
  echo "  - Ollama started with PID: $OLLAMA_PID"
else
  echo "  - Ollama already running."
fi

sleep 5

echo "[+] Starting FastAPI Vector service (VectorApp.py) on port ${API_PORT}..."
# Uses uvicorn with VectorApp.app FastAPI instance.[file:3]
nohup "$VENV_DIR/bin/uvicorn" VectorApp:app \
  --host 0.0.0.0 --port "${API_PORT}" \
  > "$PROJECT_DIR/log_fastapi.out" 2>&1 &

FASTAPI_PID=$!
echo "  - FastAPI started with PID: $FASTAPI_PID"

sleep 5

echo "[+] Starting Streamlit Admin app (AdminApp.py) on port ${STREAMLIT_PORT}..."
nohup "$VENV_DIR/bin/streamlit" run AdminApp.py \
  --server.port "${STREAMLIT_PORT}" \
  --server.address 0.0.0.0 \
  > "$PROJECT_DIR/log_streamlit.out" 2>&1 &

STREAMLIT_PID=$!
echo "  - Streamlit started with PID: $STREAMLIT_PID"

echo
echo "[+] All services started."
echo "  - Ollama API:      http://localhost:${OLLAMA_PORT}"
echo "  - FastAPI Vector:  http://localhost:${API_PORT}"
echo "  - Admin UI:        http://localhost:${STREAMLIT_PORT}"
echo
echo "[+] Logs:"
echo "  - $PROJECT_DIR/log_ollama.out"
echo "  - $PROJECT_DIR/log_fastapi.out"
echo "  - $PROJECT_DIR/log_streamlit.out"
echo
echo "[INFO] To stop services, use: kill $OLLAMA_PID $FASTAPI_PID $STREAMLIT_PID (or pkill)."
