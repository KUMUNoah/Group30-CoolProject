from anthropic import Anthropic

def format_symptoms(metadata: dict) -> str:
    symptoms = []
    if metadata.get('itch') == 1:      symptoms.append('itching')
    if metadata.get('bleed') == 1:     symptoms.append('bleeding')
    if metadata.get('grew') == 1:      symptoms.append('growing')
    if metadata.get('hurt') == 1:      symptoms.append('painful')
    if metadata.get('changed') == 1:   symptoms.append('changing appearance')
    if metadata.get('elevation') == 1: symptoms.append('elevated')

    lines = [f"Age: {metadata.get('age', 'unknown')}"]
    if symptoms:
        lines.append(f"Symptoms: {', '.join(symptoms)}")
    if metadata.get('skin_cancer_history') == 1:
        lines.append("History of skin cancer: yes")
    if metadata.get('fitspatrick'):
        lines.append(f"Fitzpatrick type: {metadata['fitspatrick']}")
    return '\n'.join(lines)

def generate_response(query: str, metadata: dict, cnn_prediction: dict, retrieved_chunks: list) -> str:
    client = Anthropic()
    
    content = f"""SkinAI, an app that helps users determine if they have skin cancer based on their symptoms, medical history and an upploaded picture
    of their skin lesion.
    
    CNN PREDICTION: {cnn_prediction["label"]} with confidence {cnn_prediction["confidence"]:.2f}
    
    Patient Symptoms: {format_symptoms(metadata)}
    Patient Description: {query}
    """
    
    for i, chunk in enumerate(retrieved_chunks):
        content += f"\n\n--Source {i+1}--\nDisease: {chunk['disease']}\nContent: {chunk['content']}\nURL: {chunk['url']}"
        
    content += """Based on the above information, provide a concise summary of the patient's condition, including:
    1. The most likely diagnosis
    2. The reasoning behind this diagnosis
    3. Any recommended next steps for the patient (e.g. see a doctor, get a biopsy, etc.)"""
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system ="You are a medical education assistatn for SkinAi. You provide educational information only not a straight diagnosis.",
        messages=[{"role": "user", "content": content}],
    )
    return message.content[0].text