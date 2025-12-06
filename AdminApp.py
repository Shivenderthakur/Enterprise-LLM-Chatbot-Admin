# app.py
# app.py
import os
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Optional, Iterator, Tuple
from urllib.parse import unquote, urlparse, quote

import streamlit as st
import requests
from dotenv import load_dotenv
load_dotenv()
# --- Safe imports with helpful messages ---
try:
    import fitz  # PyMuPDF
except Exception as e:
    st.error(
        "Missing or broken PyMuPDF (`pymupdf`). Fix with:\n\n"
        "`pip uninstall -y fitz && pip install --upgrade pymupdf`\n\n"
        f"Import error: {e}"
    )
    st.stop()

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    st.error(
        "Missing `sentence-transformers`. Install with:\n\n"
        "`pip install sentence-transformers`\n\n"
        f"Import error: {e}"
    )
    st.stop()

try:
    from pymongo import MongoClient
except Exception as e:
    st.error(
        "Missing `pymongo`. Install with:\n\n"
        "`pip install pymongo`\n\n"
        f"Import error: {e}"
    )
    st.stop()

# --------------------------- Config -----------------------------------
st.set_page_config(page_title="Ollama Admin + KB", layout="wide")

OLLAMA_BASE = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
API_URL = OLLAMA_BASE
GPU_MEMORY_LIMIT = int(os.getenv("GPU_MEMORY_LIMIT_BYTES", 14 * 1024**3))

ADMIN_USER = os.getenv("APP_ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("APP_ADMIN_PASS", "admin")

INLINE_MONGO_URI = os.getenv(
    "INLINE_MONGO_URI",
)

DB_NAME = os.getenv("KB_DB_NAME", "vector_db")
COLLECTION = os.getenv("KB_COLLECTION", "pdf_embeddings")
VECTOR_FIELD = os.getenv("VECTOR_FIELD", "embedding")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")

ADMIN_API_BASE_DEFAULT = os.getenv("ADMIN_API_BASE", "http://localhost:5000") # This will need to point to the FastAPI service's internal address

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

def infer_vector_dim(model_name: str) -> int:
    n = (model_name or "").lower()
    if "bge" in n:
        return 768
    if "e5" in n:
        return 768
    if "mini" in n or "all-minilm" in n:
        return 384
    return int(os.getenv("VECTOR_DIM", 384))

VECTOR_DIM = infer_vector_dim(EMBED_MODEL)

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------- Utilities --------------------------------
def safe_st_secrets_get(key: str, default: Optional[str] = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

def pick_mongo_uri() -> str:
    env_val = os.getenv("MONGODB_URI", "").strip()
    if env_val:
        return env_val
    secret_val = safe_st_secrets_get("MONGODB_URI", "").strip()
    if secret_val:
        return secret_val
    return INLINE_MONGO_URI

def safe_model_key(name: str) -> str:
    return hashlib.md5((name or "").encode()).hexdigest()

def format_bytes(num: int) -> str:
    if not num:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"

def one_time_error(key: str, message: str):
    if not st.session_state.get(key, False):
        st.error(message)
        st.session_state[key] = True

# --------------------------- Auth -------------------------------------
def is_authed() -> bool:
    return st.session_state.get("auth", False)

def login_view():
    st.title("🔐 Administrator Login")
    st.caption("Enter admin credentials to access the control panel.")
    user = st.text_input("Username", key="login_user")
    pwd = st.text_input("Password", type="password", key="login_pwd")
    if st.button("Login", key="login_btn"):
        if user == ADMIN_USER and pwd == ADMIN_PASS:
            st.session_state["auth"] = True
            st.success("Logged in.")
            st.rerun()
        else:
            st.error("Invalid credentials.")

def sidebar_logout():
    if st.sidebar.button("Logout", key="sidebar_logout"):
        st.session_state.clear()
        st.rerun()

# --------------------------- Mongo helpers ----------------------------
MONGO_URI = pick_mongo_uri()

@st.cache_resource(show_spinner=False)
def get_mongo_client_and_db():
    if not MONGO_URI:
        return None, None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=6000)
        db = client[DB_NAME]
        client.admin.command("ping")
        return client, db
    except Exception as e:
        one_time_error("err_mongo", f"💾 MongoDB connection failed: {e}")
        return None, None

def ensure_vector_index(db, index_name: str = VECTOR_INDEX_NAME, dims: int = VECTOR_DIM) -> bool:
    if db is None:
        return False
    coll = db[COLLECTION]
    try:
        try:
            existing = list(coll.list_search_indexes())
            have = any(ix.get("name") == index_name for ix in existing)
        except Exception:
            have = False
        if have:
            st.caption(f"Vector index '{index_name}' exists (Atlas Search Indexes).")
            return True
        st.error(
            "Atlas Vector Search index not found. Create the index BEFORE uploading embeddings."
        )
        good_def = {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "pdf_name": {"type": "string"},
                    VECTOR_FIELD: {"type": "knnVector", "dimensions": dims, "similarity": "cosine"},
                },
            }
        }
        st.markdown("**Paste this JSON into Atlas → Search Indexes (JSON editor)**")
        st.code(json.dumps({"name": index_name, "definition": good_def}, indent=2), language="json")
        st.info("Clusters → Browse Collections → Select DB & Collection → Search Indexes → Create Search Index → JSON Editor")
        return False
    except Exception as e:
        st.warning(f"Could not check Atlas search indexes: {e}")
        return False

# --------------------------- Embedding & PDF ---------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

def extract_text(file_path: str) -> str:
    doc = fitz.open(file_path)
    out = []
    for page in doc:
        txt = page.get_text()
        if txt:
            out.append(txt)
    return "\n".join(out)

def chunk_text(text: str, chunk_words: int = 500) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]

def make_chunk_id(pdf_name: str, idx: int) -> str:
    return hashlib.md5(f"{pdf_name}_{idx}".encode()).hexdigest()

def store_pdf_to_mongo(db, pdf_name: str, file_path: str, chunk_words: int = 500) -> int:
    coll = db[COLLECTION]
    text = extract_text(file_path)
    if not text.strip():
        return 0
    chunks = chunk_text(text, chunk_words)
    model = get_embedder()
    batch = int(os.getenv("EMBED_BATCH", 8))
    inserted = 0
    for i in range(0, len(chunks), batch):
        part = chunks[i:i+batch]
        vecs = model.encode(part).tolist()
        for j, (chunk_text_val, vec) in enumerate(zip(part, vecs)):
            idx = i + j
            doc = {
                "_id": make_chunk_id(pdf_name, idx),
                "pdf_name": pdf_name,
                "chunk_idx": idx,
                "chunk": chunk_text_val,
                VECTOR_FIELD: vec,
                "uploaded_at": datetime.utcnow(),
            }
            try:
                coll.insert_one(doc)
                inserted += 1
            except Exception as e:
                if "duplicate key error" in str(e).lower():
                    continue
                else:
                    st.error(f"Insert error: {e}")
                    raise
    return inserted

def list_pdfs_in_db(db) -> List[str]:
    if db is None:
        return []
    try:
        return sorted(db[COLLECTION].distinct("pdf_name"))
    except Exception:
        return []

def count_chunks(db, pdf_name: str) -> int:
    return db[COLLECTION].count_documents({"pdf_name": pdf_name})

def delete_pdf(db, pdf_name: str):
    db[COLLECTION].delete_many({"pdf_name": pdf_name})

# --------------------------- Ollama API helpers ------------------------
def _build_url(path: str) -> str:
    base = API_URL.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return base + path if path.startswith("/api") else base + "/api" + path

def api_request(path: str, method: str = "GET", data: Optional[Dict] = None, timeout: int = 60):
    url = _build_url(path)
    try:
        if method.upper() == "GET":
            res = requests.get(url, timeout=timeout)
        elif method.upper() == "POST":
            res = requests.post(url, json=data or {}, timeout=timeout)
        elif method.upper() == "DELETE":
            res = requests.delete(url, json=data or {}, timeout=timeout)
        else:
            raise ValueError("Unsupported method")
        res.raise_for_status()
        try:
            return res.json()
        except ValueError:
            return {}
    except requests.exceptions.RequestException as e:
        one_time_error("err_ollama", f"🧠 Ollama server not reachable: {e}")
        return {}

def api_request_stream(path: str, data: Optional[Dict] = None, timeout: int = 360) -> Iterator[Dict]:
    url = _build_url(path)
    try:
        with requests.post(url, json=data or {}, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    try:
                        if line.startswith("data:"):
                            yield json.loads(line[len("data:"):].strip())
                    except Exception:
                        pass
    except requests.exceptions.RequestException as e:
        one_time_error("err_stream", f"🧠 Ollama streaming error: {e}")
        return iter(())

# ---- High-level model ops ----
def get_local_models_raw() -> Dict:
    return api_request("/tags", "GET") or {}

def get_local_models() -> List[str]:
    data = get_local_models_raw()
    if isinstance(data, dict) and "models" in data:
        return [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
    if isinstance(data, list) and all(isinstance(i, dict) for i in data):
        return [d.get("name") or d.get("model") for d in data if d.get("name") or d.get("model")]
    if isinstance(data, list) and all(isinstance(i, str) for i in data):
        return data
    return []

def show_model_info(model: str) -> Dict:
    return api_request("/show", "POST", {"model": model, "verbose": True})

def pull_model_stream(model: str) -> Iterator[Dict]:
    payload = {"model": model}
    return api_request_stream("/pull", payload)

def delete_model(model: str) -> Dict:
    return api_request("/delete", "DELETE", {"model": model})

def copy_model(source: str, dest: str) -> Dict:
    return api_request("/copy", "POST", {"source": source, "destination": dest})

def create_model_from_parent(new_name: str, from_model: str, system_prompt: Optional[str] = None, quantize: Optional[str] = None) -> Iterator[Dict]:
    payload: Dict[str, Optional[str]] = {"model": new_name, "from": from_model}
    if system_prompt:
        payload["system"] = system_prompt
    if quantize:
        payload["quantize"] = quantize
    return api_request_stream("/create", payload)

def list_running_models() -> List[Dict]:
    data = api_request("/ps", "GET")
    if isinstance(data, dict) and "models" in data:
        return data.get("models", [])
    if isinstance(data, list):
        return data
    return []

def load_model_into_memory(model: str, keep_alive: Optional[str | int] = "15m") -> Dict:
    return api_request("/generate", "POST", {"model": model, "prompt": "", "stream": False, "keep_alive": keep_alive})

def unload_model_from_memory(model: str) -> Dict:
    return api_request("/generate", "POST", {"model": model, "stream": False, "keep_alive": 0})

# --------------------------- Model size helpers ------------------------
@st.cache_data(show_spinner=False)
def get_model_size(model: str) -> int:
    if not model:
        return 0
    info = show_model_info(model) or {}
    candidates = [
        info.get("size"),
        info.get("size_bytes"),
        info.get("size_vram"),
        info.get("size_vram_bytes"),
        info.get("disk_size"),
        info.get("disk_bytes"),
    ]
    for v in candidates:
        if v is None:
            continue
        if isinstance(v, dict):
            for k in ("bytes", "size_bytes", "value"):
                if k in v:
                    try:
                        return int(v[k])
                    except Exception:
                        pass
            continue
        try:
            return int(v)
        except Exception:
            s = str(v)
            digits = "".join(ch for ch in s if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except Exception:
                    pass
    details = info.get("details", {}) or {}
    for key in ("file_size", "size_bytes", "disk_bytes"):
        val = details.get(key)
        if isinstance(val, (int, str)):
            try:
                return int(val)
            except Exception:
                pass
    return 0

def is_text_gen(name: str, info: Dict) -> bool:
    n = (name or "").lower()
    blocked = ["embed", "vision", "whisper", "audio", "clip", "mm"]
    if any(tok in n for tok in blocked):
        return False
    fam = str(info.get("details", {}).get("family", "") or info.get("family", "")).lower()
    if fam and any(tok in fam for tok in ["embed", "vision"]):
        return False
    return True

def models_fitting_memory(candidates: List[str]) -> List[str]:
    out = []
    for m in candidates:
        info = show_model_info(m) or {}
        size = get_model_size(m)
        if size and size <= GPU_MEMORY_LIMIT and is_text_gen(m, info):
            out.append(m)
    return out

# --------------------------- UI Helpers --------------------------------
DEF_PAD = 6

def sidebar_quick():
    st.sidebar.header("Quick Actions")
    if st.sidebar.button("Refresh", key="sidebar_refresh"):
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption("Ollama on localhost:11434 • Vectors in MongoDB")

def render_pull_progress(events: Iterator[Dict]):
    status_box = st.empty()
    bar = st.progress(0)
    bytes_text = st.empty()
    total = 0
    completed = 0
    for ev in events:
        status = ev.get("status", "...")
        if "total" in ev:
            total = int(ev.get("total") or 0)
        if "completed" in ev:
            completed = int(ev.get("completed") or 0)
        pct = int((completed / total) * 100) if total else 0
        status_box.write(f"**{status}**")
        bar.progress(min(max(pct, 0), 100))
        if total:
            bytes_text.caption(f"{format_bytes(completed)} / {format_bytes(total)}")
    status_box.success("Done")
    bar.progress(100)

# ---------------- Admin backend helpers ----------------
if "ADMIN_API_BASE" not in st.session_state:
    st.session_state["ADMIN_API_BASE"] = ADMIN_API_BASE_DEFAULT

def get_admin_api_base() -> str:
    # Use the internal host and port for the FastAPI service
    return os.getenv("ADMIN_API_BASE", "http://localhost:8000").rstrip("/")

# ----------------- NEW: build_admin_file_url & admin_api_download_file -----------------
def build_admin_file_url(file_url_path: str) -> str:
    """
    Build canonical admin file URL:
      {ADMIN_BASE}/api/files/admin/{filename}

    - Accepts absolute URLs or relative paths.
    - Extracts filename, unquotes then re-encodes it safely.
    - Returns empty string if no filename found.
    """
    if not file_url_path:
        return ""

    # strip surrounding quotes/whitespace
    s = file_url_path.strip()
    while (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    s = s.strip('"').strip("'").strip()

    # parse to try extract filename
    try:
        parsed = urlparse(s)
        fn = os.path.basename(parsed.path) if parsed.path else ""
    except Exception:
        fn = ""

    if not fn:
        # fallback: last segment after slash or the whole string
        fn = s.split("/")[-1] if "/" in s else s

    fn = (fn or "").strip().strip('"').strip("'")
    if not fn:
        return ""

    # unquote first (avoid double-encoding), then quote fully
    try:
        fn_clean = unquote(fn)
    except Exception:
        fn_clean = fn
    fn_q = quote(fn_clean, safe="")

    base = get_admin_api_base().rstrip("/")
    return f"{base}/api/files/admin/{fn_q}"

def admin_api_download_file(file_url_path: str, dest_path: str) -> bool:
    """
    Minimal downloader that builds the canonical admin URL and downloads the file.
    Expects get_admin_api_base() to be set (via session_state or env).
    """
    import requests
    import os

    admin_url = build_admin_file_url(file_url_path)
    if not admin_url:
        st.error("⚠️ Could not determine filename from backend URL/path.")
        return False

    st.info(f"📥 Downloading (admin) from: {admin_url}")
    try:
        r = requests.get(admin_url, stream=True, timeout=60)
    except Exception as e:
        st.error(f"❌ Network error when requesting {admin_url}: {e}")
        return False

    if r.status_code != 200:
        st.error(f"❌ Failed to download: {r.status_code} {r.reason} — tried {admin_url}")
        return False

    try:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success(f"✅ Downloaded: {os.path.basename(dest_path)}")
        return True
    except Exception as e:
        st.error(f"❌ Error writing file to disk: {e}")
        return False

# ----------------- Updated normalize_file_url (kept for compatibility) --------------
def normalize_file_url(file_url_path: str) -> str:
    """
    Keep a normalized absolute URL helper for 'View' links.
    This will attempt to return the admin URL (preferred) so 'Open in new tab' goes to:
      {ADMIN_BASE}/api/files/admin/{filename}
    If filename missing, falls back to building an absolute URL using ADMIN base + original path.
    """
    if not file_url_path:
        return ""
    # Try to return admin-style URL when possible
    admin_url = build_admin_file_url(file_url_path)
    if admin_url:
        return admin_url

    # Fallback: build as absolute/encoded using admin base
    cleaned = file_url_path.strip()
    while (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.strip('"').strip("'").strip()
    try:
        cleaned = unquote(cleaned)
    except Exception:
        pass

    lower = cleaned.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        p = urlparse(cleaned)
        encoded_path = quote(p.path, safe="/")
        rebuilt = f"{p.scheme}://{p.netloc}{encoded_path}"
        if p.query:
            rebuilt += "?" + p.query
        return rebuilt

    base = get_admin_api_base()
    if not cleaned.startswith("/"):
        cleaned = "/" + cleaned
    encoded = quote(cleaned, safe="/")
    return base.rstrip("/") + encoded

def admin_api_get_file_list(status: str) -> List[Dict]:
    base = get_admin_api_base()
    url = f"{base}/api/files/admin/file-list/{status}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        resp = r.json()
        if isinstance(resp, dict) and resp.get("status") and "data" in resp:
            return resp.get("data", [])
        if isinstance(resp, list):
            return resp
        return []
    except requests.exceptions.RequestException as e:
        one_time_error("err_admin_api", f"Admin backend not reachable at {base}: {e}")
        return []

def admin_api_patch_status(file_id: str, new_status: str) -> bool:
    base = get_admin_api_base()
    url = f"{base}/api/files/admin"
    payload = {"id": file_id, "status": new_status}
    try:
        r = requests.patch(url, json=payload, timeout=15)
        r.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to update status for {file_id}: {e}")
        return False



# --------------------------- UI: Tabs ---------------------------------
def ollama_tab():
    st.subheader("🧠 Ollama — Local Models Admin Panel (server-side)")

    ver = api_request("/version", "GET")
    if isinstance(ver, dict) and "version" in ver:
        st.caption(f"Ollama version: {ver.get('version')}")
    else:
        st.caption("Ollama version: unreachable or unknown")

    st.markdown("---")

    t1, t2, t3, t4 = st.tabs([
        "📦 Installed Models",
        "▶️ Running",
        "➕ Add / Create / Copy",
        "ℹ️ Inspect",
    ])

    # Installed Models
    with t1:
        installed_raw = get_local_models_raw()
        installed = get_local_models()
        if not installed:
            st.info("No installed models found.")
        else:
            sizes = {}
            total_bytes = 0
            with st.spinner("Reading model sizes…"):
                for m in installed:
                    s = get_model_size(m)
                    sizes[m] = s
                    if isinstance(s, int):
                        total_bytes += s
            st.write(f"Total reported disk space (sum of model reports): **{format_bytes(total_bytes)}**")

            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                search = st.text_input("Filter by name", "", key="filter_name")
            with c2:
                only_fit = st.checkbox("Only that fit GPU limit", value=False)
            with c3:
                st.caption(f"GPU limit: {format_bytes(GPU_MEMORY_LIMIT)}")

            rows = []
            for m in installed:
                if search and search.lower() not in m.lower():
                    continue
                if only_fit:
                    info = show_model_info(m) or {}
                    if get_model_size(m) > GPU_MEMORY_LIMIT or not is_text_gen(m, info):
                        continue
                rows.append((m, sizes.get(m, 0)))

            for m, s in rows:
                k = safe_model_key(m)
                c1, c2, c3, c4 = st.columns([5, 2, 1, 1])
                with c1:
                    st.write(f"`{m}` — {format_bytes(s)}")
                with c2:
                    if st.button("Show", key=f"show_{k}"):
                        st.session_state["select_model"] = m
                        st.session_state["inspect_tab"] = True
                with c3:
                    if st.button("Run", key=f"run_{k}"):
                        load_model_into_memory(m)
                        st.toast(f"Load initiated for {m}")
                        st.rerun()
                with c4:
                    if st.button("Delete", key=f"del_{k}"):
                        delete_model(m)
                        st.success(f"Deleted {m}")
                        st.rerun()

            with st.expander("Raw /api/tags payload"):
                st.json(installed_raw)

    # Running Models
    with t2:
        running = list_running_models()
        if not running:
            st.info("No models currently loaded.")
        else:
            for r in running:
                name = r.get("name") or r.get("model")
                k = safe_model_key(name or "")
                size_vram = r.get("size_vram") or r.get("size")
                c1, c2, c3 = st.columns([5, 2, 1])
                with c1:
                    st.write(f"**{name}** — VRAM: {format_bytes(int(size_vram or 0))}")
                with c2:
                    expires = r.get("expires_at")
                    st.caption(f"expires: {expires}")
                with c3:
                    if st.button("Stop", key=f"stop_run_{k}"):
                        unload_model_from_memory(name)
                        st.success(f"Stopped {name}")
                        st.rerun()

            st.markdown("---")
            st.caption("Tip: models auto-unload after keep_alive (default 5m). Use Run with keep-alive below to pin longer.")

            pin_c1, pin_c2 = st.columns([2, 1])
            with pin_c1:
                to_pin = st.selectbox("Run model with custom keep_alive", [""] + get_local_models())
            with pin_c2:
                keep = st.text_input("keep_alive (e.g., 30m, -1)", "30m")
            if st.button("Run / Pin", disabled=not to_pin):
                load_model_into_memory(to_pin, keep_alive=keep)
                st.toast(f"Running {to_pin} with keep_alive={keep}")
                st.rerun()

    # Add / Create / Copy
    with t3:
        st.markdown("### Pull from Library")
        new_model = st.text_input("Model name (e.g., llama3.2 or `hf.co/owner/model`)", key="install_input")
        colA, colB = st.columns([1, 3])
        with colA:
            if st.button("Pull Model", key="install_btn"):
                if new_model.strip():
                    with st.spinner(f"Pulling {new_model.strip()}…"):
                        events = pull_model_stream(new_model.strip())
                        render_pull_progress(events)
                        st.success("Pull finished or resumed.")
                        st.rerun()
                else:
                    st.warning("Enter model name.")
        with colB:
            st.caption("Streaming progress from `/api/pull`.")

        st.divider()
        st.markdown("### Create / Copy Locally")
        with st.expander("Copy an existing local model"):
            src = st.selectbox("Source", get_local_models(), key="copy_src")
            dst = st.text_input("Destination name", key="copy_dst")
            if st.button("Copy", key="copy_btn", disabled=not (src and dst)):
                resp = copy_model(src, dst)
                if isinstance(resp, dict) and resp.get("error"):
                    st.error(resp.get("error"))
                else:
                    st.success(f"Copied {src} → {dst}")
                    st.rerun()

        with st.expander("Create a model from a parent (system prompt / quantize)"):
            nm = st.text_input("New model name", key="create_new")
            parent = st.selectbox("From model", get_local_models(), key="create_parent")
            sysmsg = st.text_area("System prompt (optional)", height=120)
            quant = st.text_input("Quantize (optional, e.g., q4_K_M)", "")
            if st.button("Create", key="create_btn", disabled=not (nm and parent)):
                events = create_model_from_parent(nm, parent, system_prompt=sysmsg or None, quantize=quant or None)
                render_pull_progress(events)
                st.success(f"Created {nm}")
                st.rerun()

        st.caption("Advanced: /api/create supports GGUF and safetensors via blob uploads. Add later if needed.")

    # Inspect
    with t4:
        default_sel = st.session_state.get("select_model") if st.session_state.get("inspect_tab") else ""
        local_models = [""] + get_local_models()
        selected_index = 0
        if default_sel in local_models:
            selected_index = local_models.index(default_sel)
        selected = st.selectbox("Choose model", local_models, index=selected_index, key="select_model")
        if selected:
            st.markdown(f"#### `{selected}`")
            info = show_model_info(selected) or {}

            c1, c2 = st.columns([1, 1])
            with c1:
                size = get_model_size(selected)
                st.metric("Reported size", format_bytes(size))
                st.json(info.get("details", {}))
            with c2:
                st.subheader("Modelfile")
                st.code(info.get("modelfile", "(not available)"), language="bash")
                st.subheader("Template")
                st.code(info.get("template", ""))

            st.markdown("---")
            running = list_running_models()
            running_names = [r.get("name") for r in running if r.get("name")]
            is_running = selected in running_names
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                if st.button("Run (default keep_alive)", key=f"run_selected_{safe_model_key(selected)}"):
                    resp = load_model_into_memory(selected)
                    if resp is not None:
                        st.success(f"Load initiated for {selected}")
                    else:
                        st.error(f"Failed to load {selected}")
                    st.rerun()
            with col2:
                ka = st.text_input("keep_alive (e.g., 10m, -1)", value="15m", key=f"ka_{safe_model_key(selected)}")
                if st.button("Run with keep_alive", key=f"run_ka_{safe_model_key(selected)}"):
                    load_model_into_memory(selected, keep_alive=ka)
                    st.toast(f"Run {selected} with keep_alive={ka}")
                    st.rerun()
            with col3:
                if st.button("Stop / Unload", key=f"stop_selected_{safe_model_key(selected)}"):
                    resp = unload_model_from_memory(selected)
                    if resp is not None:
                        st.success(f"Unload request sent for {selected}")
                    else:
                        st.error(f"Failed to unload {selected}")
                    st.rerun()

# --------------------------- KB Tab (with approvals) -----------------------------------

def kb_tab():
    st.subheader("📚 Knowledge Base Management")
    st.caption("Upload admin files and review user-submitted PDFs for approval.")

    # Admin API base override in session state (no global usage)
    with st.expander("Admin API / Backend settings", expanded=False):
        st.write("Change backend base URL for admin files if needed.")
        current_base = st.session_state.get("ADMIN_API_BASE", ADMIN_API_BASE_DEFAULT)
        new_base = st.text_input("Admin backend base URL", value=current_base, key="admin_api_base_input")
        if st.button("Apply backend URL"):
            st.session_state["ADMIN_API_BASE"] = new_base.strip()
            st.success(f"Set ADMIN API base to: {st.session_state['ADMIN_API_BASE']}")
            st.rerun()

    client, db = get_mongo_client_and_db()
    if db is None:
        st.warning("No Mongo URI configured. KB features disabled.")
        return

    try:
        client.admin.command("ping")
        st.caption("MongoDB: connected ✅")
    except Exception as e:
        st.error(f"MongoDB ping failed: {e}")
        return

    if not ensure_vector_index(db, index_name=VECTOR_INDEX_NAME, dims=VECTOR_DIM):
        st.stop()

    a1, a2, a3, a4 = st.tabs(["📤 Admin Uploads", "🧾 Approvals — Pending", "✅ Approvals — Approved", "🚫 Approvals — Cancelled"])

    # Admin Uploads
    with a1:
        st.markdown("### Admin Upload Panel")
        st.caption("Upload PDFs to create embeddings in MongoDB (admin-side).")

        with st.expander("➕ Upload PDFs (admin)", expanded=True):
            files = st.file_uploader("Select PDFs", type=["pdf"], accept_multiple_files=True, key="uploader_admin")
            chunk_words = st.number_input("Chunk size (words)", min_value=100, max_value=1500, value=500, step=50, key="chunk_words_admin")
            if files and st.button("Process & Index (Admin Uploads)", key="process_index_admin"):
                for f in files:
                    path = os.path.join(UPLOAD_DIR, f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    with st.spinner(f"Indexing {f.name} ..."):
                        try:
                            added = store_pdf_to_mongo(db, f.name, path, chunk_words)
                            if added > 0:
                                st.success(f"{f.name}: indexed {added} chunks")
                            else:
                                st.warning(f"{f.name}: no extractable text found")
                        except Exception as e:
                            st.error(f"Indexing error for {f.name}: {e}")
                st.rerun()

        st.markdown("### Stored PDFs (metadata only)")
        names = list_pdfs_in_db(db)
        if not names:
            st.info("No PDFs stored yet in the KB.")
        else:
            for name in names:
                k = safe_model_key(name)
                count = count_chunks(db, name)
                c1, c2, c3 = st.columns([5, 2, 1.5])
                with c1:
                    st.write(f"**{name}**")
                with c2:
                    st.write(f"Chunks: {count}")
                with c3:
                    if st.button("Delete", key=f"kb_del_{k}"):
                        delete_pdf(db, name)
                        st.success(f"Deleted {name} and its vectors.")
                        st.rerun()
            st.caption("Note: No PDF content preview is shown by design (privacy & policy).")

    # Approvals - Pending
    with a2:
        st.markdown("### Approvals — Pending files")
        st.caption("Review user uploads. Approve to promote to Admin Uploads (then press 'Done' to create embeddings).")

        col_f1, col_f2 = st.columns([3,1])
        with col_f1:
            show_count = st.number_input("Max rows", min_value=1, max_value=500, value=50, key="pending_max")
        with col_f2:
            if st.button("Refresh Pending"):
                st.session_state.pop("err_admin_api", None)
                st.rerun()

        pending_files = admin_api_get_file_list("pending")[:show_count]
        if not pending_files:
            st.info("No pending files returned by admin backend.")
        else:
            st.write(f"Pending files: {len(pending_files)}")
            for f in pending_files:
                fid = f.get("_id")
                fname = f.get("originalName") or f.get("filename") or "unknown.pdf"
                filesize = f.get("size", 0)
                user = f.get("userId", {}) or {}
                username = user.get("username", "unknown")
                email = user.get("email", "")
                status = f.get("status", "pending")
                created = f.get("createdAt")
                url_path = f.get("url")

                with st.container():
                    c1, c2, c3 = st.columns([4,2,2])
                    with c1:
                        st.markdown(f"**{fname}**")
                        st.caption(f"Uploaded by: {username} — {email}")
                        st.write(f"Size: {format_bytes(int(filesize))} • Uploaded: {created} • Status: `{status}`")
                    with c2:
                        if url_path:
                            # Use admin URL for view link
                            full_url = build_admin_file_url(url_path)
                            if full_url:
                                if st.button("👁 View", key=f"view_{fid}"):
                                    st.markdown(f"[Open file in new tab]({full_url})", unsafe_allow_html=True)
                    with c3:
                        if st.button("✅ Approve", key=f"approve_{fid}"):
                            ok = admin_api_patch_status(fid, "approved")
                            if ok:
                                st.success(f"Set {fname} → approved")
                                st.rerun()
                        if st.button("❌ Cancel", key=f"cancel_{fid}"):
                            ok = admin_api_patch_status(fid, "cancelled")
                            if ok:
                                st.success(f"Set {fname} → cancelled")
                                st.rerun()

            st.markdown("---")
            st.info("After approving files they will be processed into the KB when you press 'Done — Process Approved Files' below.")

            if st.button("✅ Done — Process Approved Files"):
                st.info("Processing approved files: downloading and creating embeddings...")
                approved_files = admin_api_get_file_list("approved")
                if not approved_files:
                    st.info("No files with status 'approved' found to process.")
                else:
                    processed = 0
                    failed = 0
                    for af in approved_files:
                        afid = af.get("_id")
                        afname = af.get("filename") or af.get("originalName") or f"file_{afid}.pdf"
                        af_url = af.get("url")
                        local_path = os.path.join(UPLOAD_DIR, afname)
                        st.write(f"Processing `{afname}` ...")
                        ok = admin_api_download_file(af_url, local_path)
                        if not ok:
                            st.error(f"Failed to download {afname}")
                            failed += 1
                            continue
                        try:
                            added = store_pdf_to_mongo(db, afname, local_path, chunk_words=int(st.session_state.get("chunk_words_admin", 500)))
                            if added > 0:
                                st.success(f"{afname}: indexed {added} chunks")
                                processed += 1
                            else:
                                st.warning(f"{afname}: no extractable text found")
                                failed += 1
                        except Exception as e:
                            st.error(f"Indexing error for {afname}: {e}")
                            failed += 1
                    st.success(f"Done: {processed} processed, {failed} failed.")
                    st.rerun()

    # Approvals - Approved
    with a3:
        st.markdown("### Approvals — Approved files")
        st.caption("Files already marked approved. You can process them (download + create embeddings) individually or in batch.")

        col_a1, col_a2 = st.columns([3,1])
        with col_a1:
            approved_limit = st.number_input("Max rows", min_value=1, max_value=500, value=50, key="approved_max")
        with col_a2:
            if st.button("Refresh Approved"):
                st.session_state.pop("err_admin_api", None)
                st.rerun()

        approved_files = admin_api_get_file_list("approved")[:approved_limit]
        if not approved_files:
            st.info("No approved files found.")
        else:
            st.write(f"Approved files: {len(approved_files)}")
            for f in approved_files:
                fid = f.get("_id")
                fname = f.get("originalName") or f.get("filename") or "unknown.pdf"
                filesize = f.get("size", 0)
                user = f.get("userId", {}) or {}
                username = user.get("username", "unknown")
                email = user.get("email", "")
                status = f.get("status", "approved")
                created = f.get("createdAt")
                url_path = f.get("url")

                with st.container():
                    c1, c2, c3 = st.columns([4,2,2])
                    with c1:
                        st.markdown(f"**{fname}**")
                        st.caption(f"Uploaded by: {username} — {email}")
                        st.write(f"Size: {format_bytes(int(filesize))} • Uploaded: {created} • Status: `{status}`")
                    with c2:
                        if url_path:
                            # Use admin URL for view link
                            full_url = build_admin_file_url(url_path)
                            if full_url:
                                if st.button("👁 View", key=f"a_view_{fid}"):
                                    st.markdown(f"[Open file in new tab]({full_url})", unsafe_allow_html=True)
                    with c3:
                        if st.button("⬇️ & Embed (Process)", key=f"process_{fid}"):
                            local_fname = fname
                            local_path = os.path.join(UPLOAD_DIR, local_fname)
                            st.info(f"Downloading {fname} ...")
                            ok = admin_api_download_file(url_path, local_path)
                            if not ok:
                                st.error(f"Download failed for {fname}")
                            else:
                                st.info(f"Indexing {fname} ...")
                                try:
                                    added = store_pdf_to_mongo(db, local_fname, local_path, chunk_words=int(st.session_state.get("chunk_words_admin", 500)))
                                    if added > 0:
                                        st.success(f"{fname}: indexed {added} chunks")
                                    else:
                                        st.warning(f"{fname}: no extractable text found")
                                except Exception as e:
                                    st.error(f"Indexing error for {fname}: {e}")
                            st.rerun()

            st.markdown("---")
            if st.button("⬇️ & Embed ALL Approved"):
                st.info("Downloading and indexing all approved files (batch)...")
                processed = 0
                failed = 0
                for af in approved_files:
                    afid = af.get("_id")
                    afname = af.get("filename") or af.get("originalName") or f"file_{afid}.pdf"
                    af_url = af.get("url")
                    local_path = os.path.join(UPLOAD_DIR, afname)
                    ok = admin_api_download_file(af_url, local_path)
                    if not ok:
                        st.error(f"Failed to download {afname}")
                        failed += 1
                        continue
                    try:
                        added = store_pdf_to_mongo(db, afname, local_path, chunk_words=int(st.session_state.get("chunk_words_admin", 500)))
                        if added > 0:
                            st.success(f"{afname}: indexed {added} chunks")
                            processed += 1
                        else:
                            st.warning(f"{afname}: no extractable text found")
                            failed += 1
                    except Exception as e:
                        st.error(f"Indexing error for {afname}: {e}")
                        failed += 1
                st.success(f"Done batch: {processed} processed, {failed} failed.")
                st.rerun()

    # Approvals - Cancelled
    with a4:
        st.markdown("### Approvals — Cancelled files")
        st.caption("Files that were cancelled by admin. You can move them back to pending if required.")

        col_r1, col_r2 = st.columns([3,1])
        with col_r2:
            if st.button("Refresh Cancelled"):
                st.session_state.pop("err_admin_api", None)
                st.rerun()

        cancelled_files = admin_api_get_file_list("cancelled")
        if not cancelled_files:
            st.info("No cancelled files found.")
        else:
            for f in cancelled_files:
                fid = f.get("_id")
                fname = f.get("originalName") or f.get("filename") or "unknown.pdf"
                filesize = f.get("size", 0)
                user = f.get("userId", {}) or {}
                username = user.get("username", "unknown")
                email = user.get("email", "")
                status = f.get("status", "cancelled")
                created = f.get("createdAt")
                url_path = f.get("url")

                with st.container():
                    c1, c2, c3 = st.columns([4,2,2])
                    with c1:
                        st.markdown(f"**{fname}**")
                        st.caption(f"Uploaded by: {username} — {email}")
                        st.write(f"Size: {format_bytes(int(filesize))} • Uploaded: {created} • Status: `{status}`")
                    with c2:
                        if url_path:
                            full_url = build_admin_file_url(url_path)
                            if full_url:
                                if st.button("👁 View", key=f"c_view_{fid}"):
                                    st.markdown(f"[Open file in new tab]({full_url})", unsafe_allow_html=True)
                    with c3:
                        if st.button("↩️ Move to Pending", key=f"move_pending_{fid}"):
                            ok = admin_api_patch_status(fid, "pending")
                            if ok:
                                st.success(f"Set {fname} → pending")
                                st.rerun()

# --------------------------- Main -------------------------------------

def main():
    if not is_authed():
        login_view()
        return

    sidebar_quick()
    sidebar_logout()

    tabs = st.tabs(["🧠 Ollama Admin", "📚 Knowledge Base"])
    with tabs[0]:
        ollama_tab()
    with tabs[1]:
        kb_tab()

if __name__ == "__main__":
    main()