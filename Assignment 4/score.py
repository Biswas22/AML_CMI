import mlflow
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import numpy as np
import regex as re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
import string
import pickle
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(__file__))

train = pd.read_csv("./data/train.csv")
val = pd.read_csv("./data/validation.csv")
test = pd.read_csv("./data/test.csv")

#splitting the datframe into X and y
y_train, X_train = train["y_train"], train["X_train"]
y_val, X_val = val["y_val"], val["X_val"]
y_test, X_test = test["y_test"], test["X_test"]

vectorizer = pickle.load(open('./word_vec.sav', 'rb'))

X_train_bow = vectorizer.transform(X_train)
X_val_bow = vectorizer.transform(X_val)
X_test_bow = vectorizer.transform(X_test)

print(X_train_bow.shape)
# tfidf = TfidfVectorizer()
# train_tfidf = tfidf.fit_transform(X_train_bow)
tfidf = TfidfTransformer().fit(X_train_bow)

def split_into_tokens(data):
    tokenized_words = []
    regex=r"\w+"
    
    for i in range(len(data)):
        tokenized_words.append(re.findall(regex, data))
        
    return tokenized_words

#Function to perform lematization and stopword removal
def lemmatize(data):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    
    for i in range(len(data)):
        if str(data[i]).lower() in stop_words:
            continue
            
        elif str(data[i]) in string.punctuation:
            continue
            
        else:
            lemmatized_words.append(str(lemmatizer.lemmatize(str(data[i]))).lower())             

    return lemmatized_words

def score(text:str, model, threshold:float) -> tuple:

    token_words = split_into_tokens(text)
    text = "".join(lemmatize(token_words))

    # propensity = model.predict_proba(tfidf.transform([text]))[0]
    # predictions = (float(model.predict_proba(tfidf.transform([text]))[0][0]) >= threshold)
    
    #print(tfidf.transform(vectorizer.transform([text])).shape)
    propensity = model.predict_proba(tfidf.transform(vectorizer.transform([text])))[0]
    predictions = (float(model.predict_proba(tfidf.transform(vectorizer.transform([text])))[0][1]) >= threshold)
    
    return (predictions, float(propensity[0]))
