import pickle
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import os


nbc_models = []

def load_models():
    path1 = r'Models/NBC'
    files = os.listdir(path1)
    for f in files:
        p = os.path.join(path1, f)
        m = pickle.load(open(p, 'rb'))
        nbc_models.append(m)

def nbc_model(text):
    tfidf_vectoriszer = pickle.load(open(r'TF_IDF/vectorizer.pickle', 'rb'))
    score = 0
    for m in nbc_models:
        p = m.predict(tfidf_vectoriszer.transform([text]))
        score += p[0]
        print(p)
    score /= len(nbc_models)
    print(score)
    return score

def main():
    load_models()
    review="product is good"
    nbc_score = nbc_model(review) * 100
    nbc_score = round(nbc_score, 2)
    print(nbc_score)
            
if __name__=='__main__':
    main()
    
#def classify_utterance(utt):
#        loaded_vectorizer = pickle.load(open('vectorizer1.pickle', 'rb'))
#       loaded_model = pickle.load(open('classification1.model', 'rb'))

