import streamlit as st
import os
import joblib
import zipfile
import re
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizer, BertModel
import gdown

# --- Fungsi Load Resource (dengan gdown dan cache) ---
@st.cache_resource
def load_all_resources():
    # Cek dan Unduh NLTK Stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Membuat direktori jika belum ada
    os.makedirs('models', exist_ok=True)
    os.makedirs('bert_model_demo', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # GANTI DENGAN FILE ID ANDA DARI GOOGLE DRIVE
    file_ids = {
        "rf": "1hwozTv5xtMF_M86FIStd2dE7XYx01JIM",
        "svm": "1wuGQRbl3LIEkwg93GLjiu-3u-KfzlQRN",
        "knn": "1y4BRHzMyMmk636n0hjJNLea_AozcQx_4",
        "bert_zip": "1-b0eBSKQFIJeNklm9DPk-359em0qRY0F",
        "slang": "12iZlpjKrf9bZttAIg17LEd2IkmLUDvcw"
    }

    # Path tujuan file
    paths = {
        "rf": "models/rf_model_demo.joblib",
        "svm": "models/svm_model_demo.joblib",
        "knn": "models/knn_model_demo.joblib",
        "bert_zip": "bert_model_demo.zip",
        "bert_dir": "bert_model_demo",
        "slang": "data/slang.txt"
    }

    # Download file jika belum ada menggunakan gdown
    with st.spinner("Mempersiapkan model saat pertama kali dijalankan... Ini mungkin memakan waktu beberapa menit."):
        if not os.path.exists(paths["rf"]): gdown.download(id=file_ids["rf"], output=paths["rf"], quiet=True)
        if not os.path.exists(paths["svm"]): gdown.download(id=file_ids["svm"], output=paths["svm"], quiet=True)
        if not os.path.exists(paths["knn"]): gdown.download(id=file_ids["knn"], output=paths["knn"], quiet=True)
        
        if not os.path.exists(os.path.join(paths["bert_dir"], "tokenizer")): 
            gdown.download(id=file_ids["bert_zip"], output=paths["bert_zip"], quiet=True)
            with zipfile.ZipFile(paths["bert_zip"], 'r') as zip_ref:
                zip_ref.extractall()
            os.remove(paths["bert_zip"])
        
        if not os.path.exists(paths["slang"]): gdown.download(id=file_ids["slang"], output=paths["slang"], quiet=True)

    # Load semua model dan resource
    models = {
        'Random Forest': joblib.load(paths["rf"]),
        'SVM': joblib.load(paths["svm"]),
        'KNN': joblib.load(paths["knn"])
    }
    
    tokenizer = BertTokenizer.from_pretrained(os.path.join(paths["bert_dir"], 'tokenizer'))
    bert_model = BertModel.from_pretrained(os.path.join(paths["bert_dir"], 'model'))

    normalisasi_dict = {}
    with open(paths["slang"], "r", encoding="utf-8") as file:
        for line in file:
            if ":" in line:
                f = line.strip().split(":")
                normalisasi_dict[f[0].strip()] = f[1].strip()
    
    # Kustomisasi daftar stopwords
    stopword_list = set(stopwords.words('indonesian'))
    kata_penting_untuk_dikecualikan = ["sangat", "tidak", "kurang", "suka", "bantu", "penting", "benar"]
    for kata in kata_penting_untuk_dikecualikan:
        if kata in stopword_list:
            stopword_list.remove(kata)
    indo_stopwords = stopword_list

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return models, tokenizer, bert_model, normalisasi_dict, indo_stopwords, stemmer

# --- Fungsi Preprocessing dan lainnya ---
def preprocess_text(text, normalisasi_dict, indo_stopwords, stemmer):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    for slang, baku in normalisasi_dict.items():
        pattern = r'\b' + re.escape(slang) + r'\b'
        text = re.sub(pattern, baku, text, flags=re.IGNORECASE)
    
    tokens = text.split()
    tokens = [word for word in tokens if word not in indo_stopwords]
    
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    text = ' '.join(stemmed_tokens)
    return text

def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():