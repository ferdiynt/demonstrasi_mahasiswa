import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer, BertModel
import torch
import os
import requests
import zipfile # Library untuk unzip

# --- FUNGSI UNTUK DOWNLOAD DARI GOOGLE DRIVE ---
def download_file_from_google_drive(id, destination):
    URL = f'https://drive.google.com/uc?export=download&id={id}'
    session = requests.Session()
    response = session.get(URL, stream=True)
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    progress_bar = st.progress(0, text=f"Mengunduh {os.path.basename(destination)}...")
    
    with open(destination, 'wb') as f:
        bytes_downloaded = 0
        for chunk in response.iter_content(block_size):
            if chunk:
                bytes_downloaded += len(chunk)
                f.write(chunk)
                progress_percentage = int((bytes_downloaded / total_size) * 100) if total_size > 0 else 0
                progress_bar.progress(progress_percentage, text=f"Mengunduh {os.path.basename(destination)} ({progress_percentage}%)")

    progress_bar.empty()

# --- KONFIGURASI DAN LOAD MODEL ---
@st.cache_resource
def load_resources():
    # ID File dari Google Drive Anda
    drive_files = {
        'svm_model_demo.joblib': '1wuGQRbl3LIEkwg93GLjiu-3u-KfzlQRN',
        'knn_model_demo.joblib': '1y4BRHzMyMmk636n0hjJNLea_AozcQx_4',
        'rf_model_demo.joblib': '1hwozTv5xtMF_M86FIStd2dE7XYx01JIM',
        'bert_model_demo.zip': '1DNXDvX3I7r-mqspkdnCnx4IiLinjNWUl'  # <-- ID BERT SUDAH DIPERBARUI
    }
    
    bert_path = 'bert_model_demo'

    # Download dan unzip model BERT jika folder belum ada
    if not os.path.exists(bert_path):
        zip_file_name = 'bert_model_demo.zip'
        with st.spinner(f'Mengunduh & mengekstrak {zip_file_name}... Ini hanya dilakukan sekali.'):
            download_file_from_google_drive(drive_files[zip_file_name], zip_file_name)
            with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(zip_file_name)

    # Download model ML lainnya jika belum ada
    for filename, file_id in drive_files.items():
        if filename.endswith('.joblib') and not os.path.exists(filename):
             with st.spinner(f'Mengunduh model {filename}... Ini hanya dilakukan sekali.'):
                download_file_from_google_drive(file_id, filename)

    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')

    svm_model = joblib.load('svm_model_demo.joblib')
    knn_model = joblib.load('knn_model_demo.joblib')
    rf_model = joblib.load('rf_model_demo.joblib')

    tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_path, 'tokenizer'))
    bert_model = BertModel.from_pretrained(os.path.join(bert_path, 'model'))
    bert_model.eval()

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    indo_stopwords = set(nltk.corpus.stopwords.words('indonesian'))
    
    normalisasi_dict = {}
    with open("slang.txt", "r", encoding="utf-8") as file:
        for line in file:
            if ":" in line:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    normalisasi_dict[parts[0].strip()] = parts[1].strip()

    return {
        "svm": svm_model, "knn": knn_model, "rf": rf_model,
        "tokenizer": tokenizer, "bert_model": bert_model, "stemmer": stemmer,
        "stopwords": indo_stopwords, "slang_dict": normalisasi_dict
    }

# --- FUNGSI PREPROCESSING & FITUR (TIDAK BERUBAH) ---
def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\\d+', '', text)
    text = re.sub(r'[@#]\\w+|[^\\w\\s]|[^\\x00-\\x7F]+', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def normalisasi_kata(text, slang_dict):
    if pd.isna(text): return text
    for slang, baku in slang_dict.items():
        text = re.sub(r'\\b' + re.escape(slang) + r'\\b', baku, text, flags=re.IGNORECASE)
    return text

def remove_stopwords(text, stopwords):
    return ' '.join([word for word in text.split() if word not in stopwords])

def stem_text(text, stemmer):
    return ' '.join([stemmer.stem(word) for word in text.split()])

def preprocess_text(text, resources):
    text = text.lower()
    text = clean_text(text)
    text = normalisasi_kata(text, resources["slang_dict"])
    text = remove_stopwords(text, resources["stopwords"])
    text = stem_text(text, resources["stemmer"])
    return text

def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)

# --- UI STREAMLIT ---
st.set_page_config(page_title="Analisis Sentimen", layout="wide")
st.title("🤖 Analisis Sentimen Teks Demo Mahasiswa")
st.markdown("Aplikasi ini menggunakan model Machine Learning yang dilatih dengan fitur dari IndoBERT untuk menganalisis sentimen teks terkait demonstrasi mahasiswa.")

try:
    resources = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan ID Google Drive sudah benar dan file dapat diakses oleh 'Siapa saja yang memiliki link'. Error: {e}")
    st.stop()

user_input = st.text_area("Masukkan teks untuk dianalisis di sini:", height=150, placeholder="Contoh: Aksi demo mahasiswa hari ini berjalan dengan damai dan tertib...")

if st.button("Analisis Sentimen", type="primary"):
    if user_input:
        with st.spinner('Sedang memproses dan menganalisis teks...'):
            preprocessed_text = preprocess_text(user_input, resources)
            st.subheader("Teks Setelah Preprocessing:")
            st.info(preprocessed_text)
            
            text_embedding = get_bert_embedding(preprocessed_text, resources["tokenizer"], resources["bert_model"])
            
            svm_pred = resources["svm"].predict(text_embedding)[0]
            knn_pred = resources["knn"].predict(text_embedding)[0]
            rf_pred = resources["rf"].predict(text_embedding)[0]
            
            st.subheader("Hasil Analisis Sentimen:")
            col1, col2, col3 = st.columns(3)
            
            for col, model_name, pred in [(col1, "SVM", svm_pred), (col2, "KNN", knn_pred), (col3, "Random Forest", rf_pred)]:
                with col:
                    st.metric(label=model_name, value=pred)
                    if pred == "Positive":
                        st.success("Sentimen cenderung Positif 👍")
                    else:
                        st.error("Sentimen cenderung Negatif 👎")
    else:
        st.warning("Mohon masukkan teks untuk dianalisis.")

st.sidebar.header("Tentang Aplikasi")
st.sidebar.info("Aplikasi ini adalah implementasi dari notebook analisis sentimen. Model besar dihosting di Google Drive dan diunduh saat aplikasi pertama kali dijalankan.")