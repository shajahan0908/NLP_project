# NLP_project
# Medical Diagnosis Assistant using RAG

This project is a **Medical Diagnosis Assistant** built with **Streamlit**, **FAISS**, **Sentence Transformers**, and the **Hugging Face Inference API**. It helps users get medical differential diagnoses based on symptom input, using real biomedical literature as context.

---

## Features

-  Retrieves top PubMed article abstracts relevant to the user's symptoms.
-  Uses a Language Model (e.g., Mistral-7B,LLaMA 3-8B, and GPT-3) to suggest possible diagnoses.
- Combines search (FAISS) with generation (LLM) for accurate, literature-backed suggestions.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/medical-diagnosis-assistant.git
   cd medical-diagnosis-assistant

2.Install Required Libraries
 - streamlit
 - faiss-cpu
 - sentence-transformers
 - huggingface_hub
 - numpy
   
3.Add Your PubMed Data
- Place your pubmed_results.json file in the root folder.

4.Set Hugging Face Token
- Get a token from HuggingFace.
- You can enter it directly in the sidebar of the Streamlit app.

---

## notebook is full code in python and .py file is converted into Streamlit app.
---

## Running the App

--streamlit run medical-diagnosis.py
---
