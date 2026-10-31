import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList, PointStruct

# 1. Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError(
        "Missing QDRANT_URL or QDRANT_API_KEY in environment or .env file!"
    )

# Initialize Qdrant Client (matching your HTTPS configuration)
client = QdrantClient(
    url=QDRANT_URL,
    port=443,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
)

COLLECTION_NAME = "legal_statute_chunks"

# 2. Load local embedding model
print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Dummy test data
dummy_points = [
    {
        "id": 99901,
        "text": "Section 420 IPC: Cheating and dishonestly inducing delivery of property.",
        "jurisdiction": "criminal",
        "doc_id": "statute_420",
    },
    {
        "id": 99902,
        "text": "Section 300 IPC: Murder defined, except in cases specified hereinafter.",
        "jurisdiction": "criminal",
        "doc_id": "statute_300",
    },
    {
        "id": 99903,
        "text": "Section 10 of Contract Act: All agreements are contracts if made by free consent.",
        "jurisdiction": "civil",
        "doc_id": "statute_10",
    },
]

# --- STEP 1: INGEST / UPSERT DUMMY DATA ---
print("\n--- 1. Upserting Dummy Points ---")
points_to_upsert = []
for doc in dummy_points:
    # Generate 384-dimensional vector
    vector = model.encode(doc["text"]).tolist()
    points_to_upsert.append(
        PointStruct(
            id=doc["id"],
            vector=vector,
            payload={
                "text": doc["text"],
                "jurisdiction": doc["jurisdiction"],
                "doc_id": doc["doc_id"],
            },
        )
    )

client.upsert(collection_name=COLLECTION_NAME, points=points_to_upsert)
print(f"✅ Successfully upserted {len(dummy_points)} dummy points.")

# --- STEP 2: TEST RETRIEVAL / QUERY ---
print("\n--- 2. Testing Vector Search ---")
user_query = "What legal provision deals with fraud or cheating?"
print(f"Query: '{user_query}'")

query_vector = model.encode(user_query).tolist()

search_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=2,
)

print("\nSearch Results:")
for point in search_results.points:
    print(f"  ├── ID       : {point.id}")
    print(f"  ├── Score    : {point.score:.4f}")
    print(f"  └── Text     : {point.payload['text']}")
    print("-" * 50)

# --- STEP 3: CLEANUP / DELETE DUMMY POINTS ---
print("\n--- 3. Cleaning Up Dummy Data ---")
dummy_ids = [doc["id"] for doc in dummy_points]

# Delete specific point IDs safely
client.delete(
    collection_name=COLLECTION_NAME,
    points_selector=PointIdsList(points=dummy_ids),
)
print(f"🗑️ Successfully deleted dummy point IDs: {dummy_ids}")

# Verify deletion count
remaining_count = client.count(collection_name=COLLECTION_NAME).count
print(f"📊 Remaining point count in '{COLLECTION_NAME}': {remaining_count}")