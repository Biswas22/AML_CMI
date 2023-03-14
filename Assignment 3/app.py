from flask import Flask, request, render_template, url_for, redirect
import pickle
import score
import os
import pandas as pd
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__,template_folder='template')

os.getcwd()
os.chdir('../Assignment 2/')

train = pd.read_csv("./data/train.csv")
val = pd.read_csv("./Data/validation.csv")
test = pd.read_csv("./Data/test.csv")

model_name = "Logistic Regression"
model_version = 5

model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

os.chdir('../Assignment 3/')

#splitting the datframe into X and y
y_train, X_train = train["y_train"], train["X_train"]
y_val, X_val = val["y_val"], val["X_val"]
y_test, X_test = test["y_test"], test["X_test"]

tfidf = TfidfVectorizer()
train_tfidf = tfidf.fit_transform(X_train)




threshold=0.5

@app.route('/') 
def home():
    return render_template('spam.html')


@app.route('/spam', methods=['POST'])
def spam():
    sent = request.form['sent']
    label,prop=score.score(sent,model,threshold)
    lbl="Spam" if label == 1 else "Not spam"
    ans1 = f"""The input text is {sent}"""
    ans2 = f"""The prediction is {lbl}""" 
    ans3 = f"""The propensity score is {prop}"""
    return render_template('result.html', ans1 = ans1, ans2 = ans2, ans3 = ans3)


if __name__ == '__main__': 
    app.run(debug=True)