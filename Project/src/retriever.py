from pinecone import Pinecone
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

def query_builder(metadata: dict) -> str:
    symptoms = []
    if metadata.get('itch') == 1:      symptoms.append('itching')
    if metadata.get('bleed') == 1:     symptoms.append('bleeding')
    if metadata.get('grew') == 1:      symptoms.append('growing lesion')
    if metadata.get('hurt') == 1:      symptoms.append('painful')
    if metadata.get('changed') == 1:   symptoms.append('changing appearance')
    if metadata.get('elevation') == 1: symptoms.append('elevated')

    query = f"skin lesion"
    if metadata.get('age'):
        query += f" patient age {metadata['age']}"
    if metadata.get('fitspatrick'):
        query += f" fitzpatrick type {metadata['fitspatrick']}"
    if symptoms:
        query += f" symptoms: {', '.join(symptoms)}"
    if metadata.get('skin_cancer_history') == 1:
        query += " personal history of skin cancer"
    return query

def retrieve(metadata: dict, top_k=5):
    query = query_builder(metadata)
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index('skinai-knowledge')
    # Create embedding for the query
    model = SentenceTransformer('all-mpnet-base-v2')
    query_embedding = model.encode([query])[0].tolist()
    # Retrieve top-k relevant documents
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    return [
        {'content': match['metadata']['content'], 'url': match['metadata']['url']}
        for match in results['matches']
    ]