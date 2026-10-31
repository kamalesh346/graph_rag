import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# 1. Load environment variables from .env file
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError(
        "Missing QDRANT_URL or QDRANT_API_KEY in environment or .env file!"
    )

# 2. Force port=443 and prefer_grpc=False
client = QdrantClient(
    url=QDRANT_URL,
    port=443,  # Explicitly force HTTPS port 443
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,  # Skip gRPC port 6334 lookup
)

collections_to_create = ["legal_statute_chunks", "legal_judgment_chunks"]

# 3. Create collections for all-MiniLM-L6-v2 (384 dimensions)
for col_name in collections_to_create:
    if not client.collection_exists(col_name):
        client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )
        print(f"✅ Collection '{col_name}' created successfully on Qdrant Cloud.")
    else:
        print(f"ℹ️ Collection '{col_name}' already exists.")

# 4. Verify Active Collections
cloud_collections = client.get_collections()
print(
    "\nActive Cloud Collections:",
    [c.name for c in cloud_collections.collections],
)