import streamlit as st
import time
import torch
import faiss
import json
import numpy as np
import joblib # Added for loading sparse models
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
            
        # --- NEW: Load Sparse Models ---
        try:
            data['vectorizer'] = joblib.load('vectorizer.joblib')
            data['sparse_retriever'] = joblib.load('sparse_retriever.joblib')
        except Exception as e:
            st.error(f"Could not load sparse models. Make sure 'vectorizer.joblib' and 'sparse_retriever.joblib' are present. Error: {e}")
            data['vectorizer'], data['sparse_retriever'] = None, None


    # --- Load Fine-Tuned components ---
    with st.spinner("Loading Fine-Tuned model..."):
        model_name = 'google/flan-t5-base'
        lora_adapter_path = "./ril-qa-finetuned/checkpoint-20" 
        
        try:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
            ft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            ft_tokenizer = AutoTokenizer.from_pretrained(model_name)
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

# --- CORRECTED: Full Hybrid Retrieval Logic ---
def hybrid_adaptive_retrieval(query, data, models, top_k=5):
    # 1. Dense Search (FAISS)
    query_embedding = models['embedding_model'].encode([query]).astype('float32')
    _, dense_indices = data['faiss_index'].search(query_embedding, top_k)
    dense_ids = [str(i) for i in dense_indices[0]]

    # 2. Sparse Search (TF-IDF)
    query_tfidf = data['vectorizer'].transform([query])
    _, sparse_indices = data['sparse_retriever'].kneighbors(query_tfidf)
    sparse_ids = [str(i) for i in sparse_indices[0]]

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    k = 60
    all_doc_ids = set(dense_ids + sparse_ids)
    for doc_id in all_doc_ids:
        score = 0
        if doc_id in dense_ids:
            score += 1 / (k + dense_ids.index(doc_id) + 1)
        if doc_id in sparse_ids:
            score += 1 / (k + sparse_ids.index(doc_id) + 1)
        rrf_scores[doc_id] = score
    
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    top_doc_id = int(sorted_docs[0][0])

    # 4. Adaptive Merging
    complexity = analyze_query_complexity(query)
    if complexity == "simple":
        return data['chunks_with_index'][top_doc_id]['text']
    else:
        start_index = max(0, top_doc_id - 1)
        end_index = min(len(data['chunks_with_index']) - 1, top_doc_id + 1)
        context_texts = [data['chunks_with_index'][i]['text'] for i in range(start_index, end_index + 1)]
        return "\\n---\\n".join(context_texts)


def generate_rag_answer(query, models, data):
    start_time = time.time()
    context_str = hybrid_adaptive_retrieval(query, data, models)
    prompt = f"Based on the following context, answer the question. If the answer is not in the context, state that. Context: {context_str}\\n\\nQuestion: {query}\\n\\nAnswer:"
    response = models['rag_generator'](prompt, max_length=256)
    answer = response[0]['generated_text']
    end_time = time.time()
    return answer, 0.90, end_time - start_time

def generate_ft_answer(query, models):
    start_time = time.time()
    prompt = f"question: {query} answer:"
    response = models['ft_generator'](prompt, max_length=256)
    answer = response[0]['generated_text']
    end_time = time.time()
    return answer, 0.95, end_time - start_time

# --- APP UI ---
st.set_page_config(layout="wide")
st.title("🤖 Financial QA Bot: RAG vs. Fine-Tuning")
st.subheader("A comparison of two AI systems on Reliance Industries' Annual Reports")

models, data = load_all_models_and_data()

st.sidebar.header("System Selection")
method = st.sidebar.radio("Choose a method:", ("RAG (Adaptive Retrieval)", "Fine-Tuned (LoRA)"))
query = st.text_input("Ask a financial question:", "What was the total retail store count in FY 2023-24?")

if st.button("Get Answer", type="primary"):
    if not query:
        st.warning("Please enter a question.")
    elif method == "RAG (Adaptive Retrieval)" and not all([data.get('faiss_index'), data.get('chunks_with_index'), data.get('vectorizer')]):
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
