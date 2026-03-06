import requests
from bs4 import BeautifulSoup
import json
import os

disease_data_dict = {
    'BCC': 'https://en.wikipedia.org/wiki/Basal-cell_carcinoma',
    'SCC': 'https://en.wikipedia.org/wiki/Squamous-cell_carcinoma',
    'MEL': 'https://en.wikipedia.org/wiki/Melanoma',
    'ACK': 'https://en.wikipedia.org/wiki/Actinic_keratosis',
    'SEK': 'https://en.wikipedia.org/wiki/Seborrhoeic_keratosis',
    'NEV': 'https://en.wikipedia.org/wiki/Melanocytic_nevus'
}

def load_data(url):
    headers = {'User-Agent': 'SkinAI-research-bot/1.0 (CS175 class project; skin disease education)'}
    response = requests.get(url, headers=headers)
    #print(response.text[:2000])
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.find('div', id='mw-content-text')
    #print(content)
    if not content:
        return ""
    return content.get_text(separator='')

def chunk_data(data, chunk_size = 500, overlap = 50):
    chunks = []
    words = data.split()
    start = 0
    while start < len(words):
        chunk = ' '.join(words[start:start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
        start += chunk_size - overlap
    return chunks

def jsonify_chunks(chunks, disease_name, url):
    docs = [
        {
            'disease': disease_name,
            'url': url,
            'chunk_index': i,
            'content': chunk
        }
        for i, chunk in enumerate(chunks)
    ]
    return json.dumps(docs, indent=2)

def index_diseases(file_path):
    os.makedirs(file_path, exist_ok=True)
    for disease, url in disease_data_dict.items():
        data = load_data(url)
        chunks = chunk_data(data)
        json_data = jsonify_chunks(chunks, disease, url)
        path = os.path.join(file_path, f"{disease}_data.json")
        with open(path, 'w') as f:
            f.write(json_data)
        print(f"Indexed {disease} data to {path}")
    

if __name__ == "__main__":
    RAG_DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_docs")
    index_diseases(RAG_DOCS_DIR)
    
    