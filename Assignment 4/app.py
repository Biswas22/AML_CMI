from flask import Flask, request, render_template, url_for, redirect
import pickle
import score
import os
import pandas as pd
import json
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer

os.chdir(os.path.dirname(__file__))

app = Flask(__name__,template_folder='./template')

train = pd.read_csv("./data/train.csv")
val = pd.read_csv("./data/validation.csv")
test = pd.read_csv("./data/test.csv")

model_name = "Logistic Regression"
model_version = 5

model = pickle.load(open('./lr_best_model.sav', 'rb'))

# #splitting the datframe into X and y
# y_train, X_train = train["y_train"], train["X_train"]
# y_val, X_val = val["y_val"], val["X_val"]
# y_test, X_test = test["y_test"], test["X_test"]

# tfidf = TfidfVectorizer()
# train_tfidf = tfidf.fit_transform(X_train)




threshold=0.5

@app.route('/') 
def home():
    return render_template('spam.html')


@app.route('/score', methods=['POST'])
def spam():
    sent = request.form['sent']
    label,prop=score.score(sent,model,threshold)
    lbl="Spam" if label == 1 else "Not spam"
    
    dict = {"Sentence": sent, "Prediction": lbl, "Propensity": prop}
    json_obj = json.dumps(dict, indent = 4) 

    return json_obj


if __name__ == '__main__': 
    app.run(host="0.0.0.0", port=5000, debug = True)