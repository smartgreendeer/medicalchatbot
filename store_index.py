import os
import random
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

if not os.environ.get("PINECONE_API_KEY"):
    from pinecone_notebooks import Authenticate
    Authenticate()

api_key = os.environ.get("PINECONE_API_KEY")

from pinecone import Pinecone

pc = Pinecone(api_key=api_key)

from pinecone import ServerlessSpec

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "medical-chatbot"

import time

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=spec
    )
    # wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)
time.sleep(1)

# Generate random 384-dimensional vectors
def generate_random_vector(dim):
    return [random.uniform(-1, 1) for _ in range(dim)]

upsert1 = index.upsert(
    vectors=[
        {"id": "vec1", "values": generate_random_vector(384)},
        {"id": "vec2", "values": generate_random_vector(384)},
        {"id": "vec3", "values": generate_random_vector(384)},
    ],
    namespace="ns1"
)

print(upsert1)

upsert2 = index.upsert(
    vectors=[
        {"id": "vec1", "values": generate_random_vector(384)},
        {"id": "vec2", "values": generate_random_vector(384)},
        {"id": "vec3", "values": generate_random_vector(384)},
    ],
    namespace="ns2"
)

print(upsert2)

print(index.describe_index_stats())

query_vector_ns1 = generate_random_vector(384)
query_results1 = index.query(
    namespace="ns1",
    vector=query_vector_ns1,
    top_k=3,
    include_values=True
)

print(query_results1)

query_vector_ns2 = generate_random_vector(384)
query_results2 = index.query(
    namespace="ns2",
    vector=query_vector_ns2,
    top_k=3,
    include_values=True
)

print(query_results2)

# docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)