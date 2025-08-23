import streamlit as st
import time
import torch
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from peft import PeftModel

# --- LOAD MODELS AND DATA (This runs only once thanks to the cache) ---
@st.cache_resource
def load_all_models_and_data():
    """
    Loads all necessary models and data files for the RAG and FT systems.
    This function is cached to prevent reloading on every user interaction.
    """
    models = {}
    data = {}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"Running on: **{device.upper()}**")

    # --- Load RAG components ---
    with st.spinner("Loading RAG models and data..."):
        models['embedding_model'] = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        models['rag_generator'] = pipeline('text2text-generation', model='google/flan-t5-base', device=0 if device=='cuda' else -1)
        
        try:
            data['faiss_index'] = faiss.read_index('dense_index.faiss')
        except Exception as e:
            st.error(f"Could not load FAISS index. Make sure 'dense_index.faiss' is in the same folder. Error: {e}")
            data['faiss_index'] = None

        try:
            with open('chunks_with_index.json', 'r') as f:
                data['chunks_with_index'] = json.load(f)
        except Exception as e:
            st.error(f"Could not load chunks. Make sure 'chunks_with_index.json' is in the same folder. Error: {e}")
            data['chunks_with_index'] = []

    # --- Load Fine-Tuned components ---
    with st.spinner("Loading Fine-Tuned model..."):
        model_name = 'google/flan-t5-base'
        # IMPORTANT: Update this path if your checkpoint folder has a different name
        lora_adapter_path = "./ril-qa-finetuned/checkpoint-95" 
        
        try:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
            ft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            ft_tokenizer = AutoTokenizer.from_pretrained(model_name)
            # No device arg needed here as device_map="auto" handles it
            models['ft_generator'] = pipeline('text2text-generation', model=ft_model, tokenizer=ft_tokenizer)
        except Exception as e:
            st.error(f"Could not load Fine-Tuned model from '{lora_adapter_path}'. Error: {e}")
            models['ft_generator'] = None

    return models, data

# --- RAG and FT Functions (Adapted for Streamlit) ---
def analyze_query_complexity(query):
    query = query.lower()
    complex_keywords = ["explain", "describe", "compare", "impact", "summarize", "relationship", "analyze"]
    if len(query.split()) > 8 or any(keyword in query for keyword in complex_keywords):
        return "complex"
    else:
        return "simple"

def hybrid_adaptive_retrieval(query, data, models, top_k=1):
    complexity = analyze_query_complexity(query)
    query_embedding = models['embedding_model'].encode([query]).astype('float32')
    _, indices = data['faiss_index'].search(query_embedding, top_k)
    retrieved_chunk_index = indices[0][0]

    if complexity == "simple":
        return data['chunks_with_index'][retrieved_chunk_index]['text']
    else:
        start_index = max(0, retrieved_chunk_index - 1)
        end_index = min(len(data['chunks_with_index']) - 1, retrieved_chunk_index + 1)
        context_texts = [data['chunks_with_index'][i]['text'] for i in range(start_index, end_index + 1)]
        return "\\n---\\n".join(context_texts)

def generate_rag_answer(query, models, data):
    start_time = time.time()
    financial_keywords = {'revenue', 'profit', 'ebitda', 'jio', 'retail', 'subscriber', 'reliance', 'crore', 'billion', 'financial'}
    if not any(kw in query.lower() for kw in financial_keywords):
        return "Irrelevant question", 0.1, 0.1
    context_str = hybrid_adaptive_retrieval(query, data, models)
    prompt = f"Based on the following context, answer the question. If the answer is not in the context, state that. Context: {context_str}\\n\\nQuestion: {query}\\n\\nAnswer:"
    response = models['rag_generator'](prompt, max_length=256)
    answer = response[0]['generated_text']
    end_time = time.time()
    return answer, 0.90, end_time - start_time

def generate_ft_answer(query, models):
    start_time = time.time()
    financial_keywords = {'revenue', 'profit', 'ebitda', 'jio', 'retail', 'subscriber', 'reliance', 'crore', 'billion', 'financial'}
    if not any(kw in query.lower() for kw in financial_keywords):
        return "Irrelevant question", 0.1, 0.1
    prompt = f"question: {query} answer:"
    response = models['ft_generator'](prompt, max_length=256)
    answer = response[0]['generated_text']
    refusal_phrases = ["i am a large language model", "i cannot answer", "i don't have information"]
    if any(phrase in answer.lower() for phrase in refusal_phrases):
        answer = "[Guardrail Triggered] The model was unable to provide a factual answer."
    end_time = time.time()
    return answer, 0.95, end_time - start_time

# --- APP UI ---
st.set_page_config(layout="wide")
st.title("🤖 Financial QA Bot: RAG vs. Fine-Tuning")
st.subheader("A comparison of two AI systems on Reliance Industries' Annual Reports")

# Load models and data
models, data = load_all_models_and_data()

st.sidebar.header("System Selection")
method = st.sidebar.radio("Choose a method:", ("RAG (Adaptive Retrieval)", "Fine-Tuned (LoRA)"))
query = st.text_input("Ask a financial question:", "What was the total retail store count in FY 2023-24?")

if st.button("Get Answer", type="primary"):
    if not query:
        st.warning("Please enter a question.")
    elif method == "RAG (Adaptive Retrieval)" and (not data.get('faiss_index') or not data.get('chunks_with_index')):
        st.error("RAG components not loaded. Cannot proceed.")
    elif method == "Fine-Tuned (LoRA)" and not models.get('ft_generator'):
        st.error("Fine-Tuned model not loaded. Cannot proceed.")
    else:
        if method == "RAG (Adaptive Retrieval)":
            with st.spinner("Running RAG pipeline... (Retrieving > Generating)"):
                answer, confidence, response_time = generate_rag_answer(query, models, data)
        else: # Fine-Tuned
            with st.spinner("Running Fine-Tuned model..."):
                answer, confidence, response_time = generate_ft_answer(query, models)
        
        st.success(f"**Answer:** {answer}")
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Method Used", method.split(" ")[0])
        col2.metric("Confidence (Placeholder)", f"{confidence:.2f}")
        col3.metric("Response Time (s)", f"{response_time:.2f}")
