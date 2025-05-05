import streamlit as st
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ===========================
# Configuration
# ===========================
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # Change this to llama, deepseek, etc.
HF_TOKEN = ""  # HuggingFace Token
PUBMED_FILE = "pubmed_results.json"

# ===========================
# Load Articles
# ===========================
@st.cache_data

def load_articles(file_path=PUBMED_FILE):
    with open(file_path, "r") as f:
        articles = json.load(f)
    return articles

# ===========================
# Build FAISS Index
# ===========================
@st.cache_resource

def build_faiss_index(articles):
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    abstracts = [article["abstract"] for article in articles]
    embeddings = model.encode(abstracts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return model, index, abstracts

# ===========================
# Search Top-K Articles
# ===========================
def search(query, model, index, abstracts, articles, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []
    for idx in indices[0]:
        if idx < len(articles):
            results.append(articles[idx])
    return results

# ===========================
# Generate Final Answer using LLM
# ===========================
def generate_answer(query, docs, model_name=LLM_MODEL, token=HF_TOKEN):
    client = InferenceClient(model_name, token=token)

    context = "\n\n".join([doc.get("abstract", "") for doc in docs])

    prompt = f"""You are a medical assistant.\n
Use the following medical literature to suggest possible diagnoses.\n
Question:\n{query}\n
Context:\n{context}\n
Based on the above, provide:\n1. Top 3 differential diagnoses\n2. Supporting evidence from retrieved documents\n3. Suggested next diagnostic steps"""

    response = client.text_generation(prompt, max_new_tokens=500, temperature=0.5)
    return response

# ===========================
# Streamlit App
# ===========================
st.set_page_config(page_title="🏥 Medical Diagnosis Assistant", page_icon="🏥", layout="wide")
st.title("🏥 Medical Diagnosis Assistant")

st.write("Enter your symptoms below. The app will retrieve related medical articles and generate possible diagnoses.")

query = st.text_input("Enter symptoms (e.g., weight loss, night sweats, persistent cough)")

if query:
    with st.spinner("Retrieving relevant medical documents..."):
        articles = load_articles()
        embed_model, faiss_index, abstracts = build_faiss_index(articles)
        retrieved_docs = search(query, embed_model, faiss_index, abstracts, articles)

    if retrieved_docs:
        st.success(f"Retrieved {len(retrieved_docs)} related articles!")

        with st.spinner("Generating medical diagnosis suggestions..."):
            final_response = generate_answer(query, retrieved_docs)

        st.subheader("\ud83d\udcc4 Final Diagnosis Suggestions:")
        st.write(final_response)
    else:
        st.warning("No relevant documents found. Please try different symptoms.")

st.sidebar.title("Settings")
st.sidebar.text_input("LLM Model Name", value=LLM_MODEL, key="model_name")
st.sidebar.text_input("HuggingFace Token", value=HF_TOKEN, type="password", key="token")

st.sidebar.title("About")
st.sidebar.info("Built with FAISS, Sentence Transformers, and HuggingFace Inference API for RAG.")