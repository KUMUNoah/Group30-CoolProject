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
    
    content = f"""SkinAI, an app that helps users determine if they have skin cancer based on their symptoms, medical history and an uploaded picture
    of their skin lesion.
    
    CNN PREDICTION: {cnn_prediction["label"]} with confidence {cnn_prediction["confidence"]:.2f}
    
    Patient Symptoms: {format_symptoms(metadata)}
    Patient Description: {query}
    """
    
    for i, chunk in enumerate(retrieved_chunks):
        content += f"\n\n--Source {i+1}--\nDisease: {chunk['disease']}\nContent: {chunk['content']}\nURL: {chunk['url']}"
        
    content += """

---

Using all of the information above, write a detailed educational report for the patient. Structure your response with the following sections:

## Overview
Briefly summarize what the CNN model predicted and what the retrieved medical knowledge suggests. Note any agreement or disagreement between the two.

## Most Likely Condition
Name the most likely condition and explain in plain language what it is — its appearance, who typically gets it, and why it develops.

## Why This May Apply to You
Cross-reference the patient's specific symptoms, age, Fitzpatrick skin type, and personal history with what is known about this condition. Explain which symptoms and risk factors align and which (if any) do not fit.

## Other Conditions to Be Aware Of
List 1-2 alternative conditions that could explain the symptoms based on the retrieved sources. For each, briefly explain why it is a possibility and how it differs from the most likely diagnosis.

## Warning Signs to Watch For
List specific changes or symptoms the patient should monitor that would indicate the condition is worsening or becoming more urgent.

## Recommended Next Steps
Provide clear, actionable guidance: whether to seek urgent care, schedule a routine dermatologist visit, monitor at home, or pursue a biopsy. Tailor this to the severity suggested by the symptoms and CNN confidence.

## Important Disclaimer
Remind the patient that this is educational information only and is not a medical diagnosis. A licensed dermatologist must evaluate any concerning skin lesion."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="You are a medical education assistant for SkinAI. You provide detailed, well-structured educational information to help patients understand their skin lesion. You do not provide a direct diagnosis. Always be thorough, clear, and compassionate.",
        messages=[{"role": "user", "content": content}],
    )
    return message.content[0].text