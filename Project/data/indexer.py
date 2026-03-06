import os
import json
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def load_json_data(dir):
    json_data = []
    for filepath in os.listdir(dir):
        if filepath.endswith('.json'):
            with open(os.path.join(dir, filepath), 'r') as f:
                data = json.load(f)
                json_data.extend(data)
    return json_data

def create_embeddings(data):
    model = SentenceTransformer('all-mpnet-base-v2')
    texts = [doc['content'] for doc in data]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    for doc, emb in zip(data, embeddings):
        doc['embedding'] = emb.tolist()  # Convert numpy array to list for JSON serialization
    return data

def index_embeddings(data):
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index('skinai-knowledge')
    for doc in data:
        index.upsert(vectors=[{
            'id': f"{doc['disease']}_{doc['chunk_index']}",
            'values': doc['embedding'],
            'metadata': {
                'disease': doc['disease'],
                'url': doc['url'],
                'content': doc['content']
            }
        }])

if __name__ == "__main__":
    RAG_DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag_docs')
    json_data = load_json_data(RAG_DOCS_DIR)
    data_with_embeddings = create_embeddings(json_data)
    index_embeddings(data_with_embeddings)
    print("Data indexed successfully!")
    