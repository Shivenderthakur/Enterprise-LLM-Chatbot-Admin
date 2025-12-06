# push_or_update_model.py
# Patch-style update for a single Model document in MongoDB Atlas

from datetime import datetime
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
import socket
import threading,os
load_dotenv()
def get_local_ip_address():
    """
    Retrieves the local IP address of the machine.
    """
    try:
        # Create a socket and connect to an external address (e.g., Google's DNS server)
        # This connection is not established to send data, but to determine the
        # local IP address used for outbound connections.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        return f"Error retrieving
ip = get_local_ip_address()
# --- MongoDB Connection URI ---

MONGO_URI = URI_DB

def run_streamlit(STREAMLIT_PORT=os.getenv("STREAMLIT_PORT"):
    
    os.system(f"streamlit run AdminApp.py --server.port {STREAMLIT_PORT}")
    thread = threading.Thread(target=run_streamlit)
    thread.start()
    return thread
def run_api(API_PORT=os.getenv("API_PORT")):

    os.system(f"uvicorn front:app --host {ip} --port {API_PORT}")

    # Start API in a separate thread
    thread = threading.Thread(target=run_api)
    thread.start()
    return thread


# --- Incoming (optional) values, mimic PATCH behavior ---



def patch_model(model_link=None, vector_link=None):
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command("ping")
        print("✅ Connected to MongoDB Atlas")

        # ✅ FIX: Explicitly check for None instead of using `or`
        db = client.get_default_database()
        if db is None:
            db = client["LLM_CHATBOT"]

        collection = db["models"]

        # Build $set patch like Express .patch()
        set_fields = {"updatedAt": datetime.utcnow()}
        if model_link is not None:
            set_fields["modelLink"] = model_link
        if vector_link is not None:
            set_fields["vectorLink"] = vector_link

        update_doc = {
            "$set": set_fields,
            "$setOnInsert": {"createdAt": datetime.utcnow()},
        }

        updated = collection.find_one_and_update(
            filter={},                 # the single "Model" doc
            update=update_doc,
            upsert=True,
            return_document=ReturnDocument.AFTER
        )

        print("✅ Model patched successfully")
        print(updated)

    except PyMongoError as e:
        print("❌ MongoDB Error:", e)
    except Exception as e:
        print("❌ Unexpected Error:", e)

if  __name__ == "__main__":

    admin_thread = run_streamlit()
    vector_thread = run_api()
    while True:
        if not admin_thread.is_alive():
            admin_thread = run_streamlit()
        if not vector_thread.is_alive():
            vector_thread = run_api()

        STREAMLIT_PORT = os.getenv("STREAMLIT_PORT")
        API_PORT = os.getenv("API_PORT")
        modelLink = f"http://{ip}:{STREAMLIT_PORT}"
        vectorLink = f"http://{ip}:{API_PORT}"
        patch_model(model_link=modelLink, vector_link=vectorLink)


