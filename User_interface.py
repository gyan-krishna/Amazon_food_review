import pickle
import streamlit as st
#from PIL import Image
import pandas as pd
#import numpy as np
import plotly.express as px
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nbc_models = []
svm_models = []
lr_models  = []
rf_models  = []

def load_models():
    path1 = r'Models/NBC'
    path2 = r'Models/LR'
    path3 = r'Models/SVM'
    path4 = r'Models/RF'
    
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

    files = os.listdir(path3)
    for f in files:
        p = os.path.join(path3, f)
        m = pickle.load(open(p, 'rb'))
        svm_models.append(m)
        
    files = os.listdir(path4)
    for f in files:
        p = os.path.join(path4, f)
        m = pickle.load(open(p, 'rb'))
        rf_models.append(m)
        
def nbc_model(text):
    tfidf_vectoriszer = pickle.load(open(r'TF_IDF/vectorizer.pickle', 'rb'))
    score = 0
    for m in nbc_models:
        #print(type(tfidf_vectoriszer.transform([text])))
        vctr = tfidf_vectoriszer.transform([text])
        p = m.predict(vctr)
        score += p[0]
    score /= len(nbc_models)
    #(score)
    return score

def lr_model(text):
    tfidf_vectoriszer = pickle.load(open(r'TF_IDF/vectorizer.pickle', 'rb'))
    score = 0
    for m in lr_models:
        p = m.predict(tfidf_vectoriszer.transform([text]))
        score += p[0]
    score /= len(lr_models)
    #(score)
    return score

def svm_model(text):
    tfidf_vectoriszer = pickle.load(open(r'TF_IDF/vectorizer.pickle', 'rb'))
    score = 0
    for m in svm_models:
        p = m.predict(tfidf_vectoriszer.transform([text]))
        score += p[0]
    score /= len(lr_models)
    #(score)
    return score

def rf_model(text):
    tfidf_vectoriszer = pickle.load(open(r'TF_IDF/vectorizer.pickle', 'rb'))
    score = 0
    for m in rf_models:
        p = m.predict(tfidf_vectoriszer.transform([text]))
        score += p[0]
    score /= len(rf_models)
    #(score)
    return score


################################################################
HEADERS = ({'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/90.0.4430.212 Safari/537.36',
            'Accept-Language': 'en-US, en;q=0.5'})

def getdata(url):
    r = requests.get(url, headers=HEADERS)
    return r.text
  
def html_code(url):
    htmldata = getdata(url)
    soup = BeautifulSoup(htmldata, 'html.parser')
    return (soup)

def pre_process_stars(star_str):
    r = star_str[0:3]
    return float(r)

def pre_process_dates(s):
    st = s[24:]
    return st 

def get_rev_zones(soup):
    review_cards = []
    for item in soup.find_all("div", class_="a-section review aok-relative"):
        review_cards.append(item)
    #for i in review_cards:
    #    (i, end = '\n-----------------------------------\n')
    return review_cards

def extract_info(card):
    if(card is None):
        return None
    name = (card.find('span', class_='a-profile-name')).getText()
    star = pre_process_stars((card.find('span', class_='a-icon-alt')).getText())
    summary = (card.find('a', class_='a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold')).getText()
    date = pre_process_dates(card.find('span', class_='a-size-base a-color-secondary review-date').getText())
    review = (card.find('div', class_='a-expander-content reviewText review-text-content a-expander-partial-collapse-content')).getText()
    helpfulness = card.find('span', class_='a-size-base a-color-tertiary cr-vote-text')
    if helpfulness:
        helpfulness = helpfulness.getText()
    else:
        helpfulness = 0
        
    #(name, star, summary, date, review, helpfulness, end = '\n-----------\n')
    return [name, star, summary, date, review, helpfulness]

def get_review_df(url):
    soup = html_code(url)
    cards = get_rev_zones(soup)
    data = []
    for i in cards:
        info = extract_info(i)
        if(info is not None):
            data.append(info)
    #(data)
    df = pd.DataFrame(data, columns = ['Name', 'Stars', 'Summary', 'date', 'review', 'helpfulness'])
    return df    

def get_details(url):
    soup = html_code(url)
    
    name = soup.find('span', class_='a-size-large product-title-word-break').getText()
    price = soup.find('span', class_='a-price-whole').getText()
    image = soup.find_all("img", class_ = "a-dynamic-image a-stretch-horizontal")
    
    if(len(image) > 0):
        image = image[0]['src']
    else:
        image = soup.find_all("img", class_ = "a-dynamic-image a-stretch-vertical")
        if(len(image) > 0):
            image = image[0]['src']
        else:
            image = ""
    
    print('product name = ', name)
    print('product cost = ', price)
    print('image = ', image)
    return [name, price, image]

def df_scoring(df):
    score = [0,0,0,0]
    print("\n\n\n",df.columns)
    print(df, '\n\n\n')
    for review in df['review']:
        score[0] += lr_model(review) * 100
        score[1] += svm_model(review) * 100
        score[2] += nbc_model(review) * 100
        score[3] += rf_model(review) * 100
        
    score[0] /= len(df['review'])
    score[1] /= len(df['review'])
    score[2] /= len(df['review'])
    score[3] /= len(df['review'])
    return score
    
########################################################################
stopwords= set(['br','the','i','me','myself','we','my','our','ours','ourselves','you',"you're","you've",\
                "you'll","you'd","your",'yours','yourself','yourselves','he','him','his','himself',\
                'she',"she's",'her','hers','herself','it',"it's",'its','itself','they','them','their',\
                'theirs','themselves','what','which','who','whom','this','that',"that'll",'these','those',\
                'am','is','are','was','were','be','been','being','have','has','had','having','do','does',\
                'did','doing','a','an','the','and','but','if','or','because','as','until','while','of',\
                'at','by','for','with','about','against','between','into','through','during','before','after',\
                'above','below','to','from','up','down','in','out','on','off','over','under','again','further',\
                'then','once','here','there','when','where','why','how','all','any','both','each','few','more',\
                'most','other','some','such','only','own','same','so','than','too','very',\
                's','t','can','will','just','don','dont','should',"should've",'now','d','ll','m','o','re',\
                've','y','ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',\
                "mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",\
                'won',"won't",'wouldn',"wouldn't"])

import re
def decontracted(phrase):
  phrase = re.sub(r"won't", "will not",phrase)
  phrase = re.sub(r"can\'t","can not",phrase)
  
  phrase = re.sub(r"n\'t"," not",phrase)
  phrase = re.sub(r"\'re"," are",phrase)
  phrase = re.sub(r"\'s"," is",phrase)
  phrase = re.sub(r"\'d"," would",phrase)
  phrase = re.sub(r"\'ll"," will",phrase)
  phrase = re.sub(r"\'t"," not",phrase)
  phrase = re.sub(r"\'ve"," have",phrase)
  phrase = re.sub(r"\'m"," am",phrase)

  return phrase

def remove_html_tags(sentance):
  clean=re.compile('<.*?>')
  sentance=re.sub(clean,' ',sentance)
  return sentance

def preprocess(sentance):
  sentance = sentance.replace('\n',' ')
  sentance = sentance.replace('\t',' ')
  sentance = re.sub(r"http\S+"," ",sentance)
  #sentance = BeautifulSoup(sentance,'lxml').get_text()
  sentance = decontracted(sentance)
  sentance = re.sub("\S*\d\S*", " ",sentance).strip()
  sentance = re.sub('[^A-Za-z]+',' ',sentance)
  sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
  stemmer = nltk.porter.PorterStemmer()
  sentance = ' '.join([stemmer.stem(token) for token in sentance.split()])
  return sentance


########################################################################
def main():
    load_models()
    st.title(":green[Amazon Food Review Analyser]")
    st.markdown("""---""")
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Amazon_logo.svg/2560px-Amazon_logo.svg.png'
    st.image(url)
    st.markdown("""---""")

    st.header(":green[Interactive Demo]")
    
    review = st.text_input("Enter review / URL :", "")
    nbc_score = 0.0
    svm_score = 0.0
    lr_score = 0.0
    rf_score = 0.0
    
    if(st.button("Predict")):
        if('http' in review):
            
            df = get_review_df(review)
            lr_score, svm_score, nbc_score, rf_score = df_scoring(df)
            
            product_name, price, image_url = get_details(review)
                
            ##----------------------------------------------------------------
            df['review'] = df['review'].apply(remove_html_tags)
            df['review'] = df['review'].apply(decontracted)
            df['review'] = df['review'].apply(preprocess)
            
            #product_name = "dummy"
            review_count = len(df['review'])
            #image_url = "https://m.media-amazon.com/images/I/61tMn7E8CwL._SX679_.jpg"
            #price = 200
            print(df['review'])
            col1, col2 = st.columns(2)
            col1.markdown("")
            col1.markdown("##### *:blue[Product] &nbsp; &nbsp; &nbsp; &nbsp; : "+product_name[:20]+"*")
            col1.markdown("##### *:blue[Reviews] &nbsp; &nbsp; &nbsp; : "+str(review_count)+"*")
            col1.markdown("##### *:blue[Price] &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; : "+str(price)+" Rs*")
            col2.image(image_url, width=250)

            
        else:
            #this product is amazing! we loved it!
            nbc_score = nbc_model(review) * 100
            nbc_score = round(nbc_score, 2)
            #(nbc_score)
            
            lr_score = lr_model(review) * 100
            lr_score = round(lr_score, 2)
            
            svm_score = svm_model(review) * 100
            svm_score = round(lr_score, 2)
            
            rf_score = rf_model(review) * 100
            rf_score = round(lr_score, 2)
        
        score = nbc_score
        col1, col2 = st.columns(2)
        if(score >= 80):
            col1.markdown("# Positive")
            col2.image('Images/5.png', width=100)
        elif(score >= 60 and score < 80):
            col1.markdown("# Moderately positive")
            col2.image('Images/4.png', width=100)
        elif(score >= 40 and score < 60):
            col1.markdown("# Neutral")
            col2.image('Images/3.png', width=100)           
        elif(score >= 20 and score < 40):
            col1.markdown("# Moderately Negative")
            col2.image('Images/2.png', width=100) 
        else:
            col1.markdown("# Negative")
            col2.image('Images/1.png', width=100)             

        st.markdown("#### Review Analysis Model")
        st.markdown("##### :red[Prediction shown below is average of six binary sub models]")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Multinomial Naive Bayes", str(nbc_score) + "%")
        col2.metric("Support Vector Machine", str(svm_score) + "%")
        col3.metric("Logistic Regression", str(lr_score) + "%")
        col4.metric("Random Forest", str(rf_score) + "%")
        
        #result = predictions(review)
        
    st.markdown("""---""")
    st.header(":green[Model Accuracy Metrics]")
    #chart_data = pd.DataFrame([['NBC:', 95],['SVM:', 90],['LR:', 85],['RF:', 88]])
    #st.bar_chart(chart_data)   
    
    df = pd.DataFrame(
    [  
         ["Multinomial Naive Bayes", 0.88,    0.89,     0.88], 
         ["Support Vector Machine", 0.87,  0.87,  0.87], 
         ["Logistic Regression", 0.88,  0.88,  0.88], 
         ["Random Forest", 0.83,  0.83,  0.83]],
    columns=["Model", "Precision", "recall", "F1-Score"]
    )

    fig = px.bar(df, x="Model", y=["Precision", "recall", "F1-Score"], barmode='group', height=400)
    # st.dataframe(df) # if need to display dataframe
    st.plotly_chart(fig)

if __name__=='__main__':
    main()
    
#def classify_utterance(utt):
#        loaded_vectorizer = pickle.load(open('vectorizer1.pickle', 'rb'))
#       loaded_model = pickle.load(open('classification1.model', 'rb'))

