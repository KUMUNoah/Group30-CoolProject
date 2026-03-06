from pinecone import Pinecone
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

_model = None
_index = None

def _get_resources():
    global _model, _index
    if _model is None:
        _model = SentenceTransformer('all-mpnet-base-v2')
    if _index is None:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        _index = pc.Index('skinai-knowledge')
    return _model, _index


def query_builder(metadata: dict, symptoms_description: str) -> str:
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
    if symptoms_description:
        query += f" Patient description: {symptoms_description}"
    
    return query

def retrieve(metadata: dict, symptoms_description: str, top_k=5):
    query = query_builder(metadata, symptoms_description)
    # Initialize Pinecone client
    model, index = _get_resources()
    # Create embedding for the query
    query_embedding = model.encode([query])[0].tolist()
    # Retrieve top-k relevant documents
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    return [
        {'content': match['metadata']['content'], 'url': match['metadata']['url'], 'disease': match['metadata']['disease']}
        for match in results['matches']
    ]