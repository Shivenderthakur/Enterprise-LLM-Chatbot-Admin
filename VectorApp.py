
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
# ------------------ FastAPI App ------------------
app = FastAPI(title="Vector Chunk API")

# ------------------ MongoDB & Embedder ------------------
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
mongo_client = MongoClient(
    os.getenv("VECTOR_DB")
)
collection = mongo_client["vector_db"]["pdf_embeddings"]

# ------------------ Request/Response Models ------------------
class ChunkRequest(BaseModel):
    text: str
    pdf_filter: Optional[str] = "All PDFs"

class ChunkResponse(BaseModel):
    chunk: str

# ------------------ Vector Search Function ------------------
def get_first_chunk(query: str, pdf_filter: Optional[str] = None) -> str:
    query_vector = embedder.encode(query).tolist()
    filter_query = {"pdf_name": pdf_filter} if pdf_filter and pdf_filter != "All PDFs" else {}

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 1,  # Only first chunk
                "filter": filter_query,
            }
        },
        {"$project": {"chunk": 1, "_id": 0}},
    ]

    results = list(collection.aggregate(pipeline))
    if results:
        return results[0]["chunk"]
    else:
        return ""

# ------------------ API Endpoint ------------------
@app.post("/embed", response_model=ChunkResponse)
def embed_text(req: ChunkRequest):
    try:
        chunk = get_first_chunk(req.text, req.pdf_filter)
        return ChunkResponse(chunk=chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

