---
layout: default
title: Proposal
---

# Project Title  
**Private Health Education Assistant Using Image Classification and RAG**

## Summary of Project  
The goal of this project is to build a privacy-focused health education assistant that provides **general medical information** based on user inputs. The system accepts either:

- A textual description of symptoms, and/or  
- An uploaded medical image (initially scoped to skin-related concerns)

The assistant produces an **educational summary** that includes:
- Broad condition categories  
- Relevant medical background information  
- Guidance on when professional medical care should be sought  

This project explores the integration of **computer vision**, **retrieval-augmented generation (RAG)**, and **multi-step agentic reasoning** to support AI-assisted health education while maintaining strong privacy considerations.

## Project Goals

### Minimum Goal  
Build a working pipeline that:
- Classifies medical images or text-only symptom queries into broad condition categories  
- Retrieves relevant educational information from a medical knowledge base using a RAG approach  

### Realistic Goal  
Extend the system into an **agentic RAG pipeline** that:
- Performs multi-step retrieval (e.g., risk factors, red flags, and when to seek care)  
- Produces structured, citation-backed educational responses  
- Includes explicit safety disclaimers  

### Moonshot Goal  
Incorporate:
- Uncertainty-aware reasoning  
- Visual explainability (e.g., segmentation or attention heatmaps)  
- A web-based interface supporting both image and text queries  

## AI Algorithms  
The system will leverage:
- **Convolutional Neural Networks (CNNs)** for medical image classification  
- **Vector embeddings and similarity search** for document retrieval  
- **Large Language Models (LLMs)** via RAG for response generation  
- An **agentic control loop** for multi-step information gathering and synthesis  

## Evaluation Plan

### Quantitative Evaluation  
The image classification component will be evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  

Evaluation will be performed on a held-out test set from selected medical imaging datasets.

### Qualitative Evaluation  
We will conduct case studies on representative symptom queries and medical images by:
- Visualizing retrieved document chunks  
- Inspecting generated responses  
- Analyzing

## AI Tool Usage
We plan to use different tools at all stages of our project but the main ones that will guide our design is:
-Pytorch for the deep network framework
-Supabase for our database
-Pinecone for our vecotr database
-Either gemini or claude for our llm api calls