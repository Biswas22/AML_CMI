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
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
import warnings
warnings.filterwarnings("ignore")

os.chdir('../Assignment 2/')
train = pd.read_csv("./data/train.csv")
val = pd.read_csv("./Data/validation.csv")
test = pd.read_csv("./Data/test.csv")
os.chdir('../Assignment 3/')

#splitting the datframe into X and y
y_train, X_train = train["y_train"], train["X_train"]
y_val, X_val = val["y_val"], val["X_val"]
y_test, X_test = test["y_test"], test["X_test"]

tfidf = TfidfVectorizer()
train_tfidf = tfidf.fit_transform(X_train)

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

    propensity = model.predict_proba(tfidf.transform([text]))[0]
    predictions = (float(model.predict_proba(tfidf.transform([text]))[0][0]) >= threshold)
    
    return (predictions, float(propensity[0]))
