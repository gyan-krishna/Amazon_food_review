import pickle
#from PIL import Image
import pandas as pd
#import numpy as np
import plotly.express as px
import os

nbc_models = []
svm_models = []
lr_models  = []
rf_models  = []

def load_models():
    path1 = r'C:\Users\gyank\OneDrive\Desktop\github\Amazon_food_review\Models\NBC'
    path2 = r'C:\Users\gyank\OneDrive\Desktop\github\Amazon_food_review\Models\LR'
    
    files = os.listdir(path1)
    for f in files:
        p = os.path.join(path1, f)
        m = pickle.load(open(p, 'rb'))
        nbc_models.append(m)
    
    files = os.listdir(path2)
    for f in files:
        p = os.path.join(path2, f)
        m = pickle.load(open(p, 'rb'))
        lr_models.append(m)

def nbc_model(text): 
    tfidf_vectoriszer = pickle.load(open(r'TF_IDF/vectorizer.pickle', 'rb'))
    score = 0
    for m in nbc_models:
        
        vctr = tfidf_vectoriszer.transform([text])
        print(vctr.shape)
        p = m.predict(vctr)
        score += p[0]
    score /= len(nbc_models)
    print(score)
    return score

def lr_model(text):
    tfidf_vectoriszer = pickle.load(open(r'TF_IDF/vectorizer.pickle', 'rb'))
    score = 0
    for m in lr_models:
        p = m.predict(tfidf_vectoriszer.transform([text]))
        score += p[0]
    score /= len(lr_models)
    print(score)
    return score

def main():
    print(os.getcwd())
    load_models()
    nbc_model("Hello world")

if __name__=='__main__':
    main()
    
#def classify_utterance(utt):
#        loaded_vectorizer = pickle.load(open('vectorizer1.pickle', 'rb'))
#       loaded_model = pickle.load(open('classification1.model', 'rb'))

