import score
import pickle
import numpy as np
import os
import requests
import time
import unittest
import mlflow
from multiprocessing import Process
from app import app

os.getcwd()
os.chdir('../Assignment 2/')

model_name = "Logistic Regression"
model_version = 5

model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

os.chdir('../Assignment 3/')

text = "Television is the opium of the masses."
obv_ham_text = "The Sun rises from the East."
obv_spam_text = "Press this link to win an aeroplane for free."
threshold = 0.55

# Defining Unit Tests

# Smoke Test: Checking if score function returns values without crashing
def test_smoke(text = text, threshold = threshold, model = model) -> None:
    label, prop = score.score(text, model, threshold)

    assert label != None
    assert prop != None

# Format Test: Check if the data type of input and output meets the requirements
def test_input_formats(text = text, threshold = threshold, model = model) -> None:
    label, prop = score.score(text, model, threshold)

    assert type(text) == str
    assert type(threshold) == float 
    assert type(label) == bool
    assert type(prop) == float 

# Prediction Value Test: Checking if the predicted value is boolean
def test_pred_value(text = obv_ham_text, threshold = threshold, model = model) -> None:
    label, prop = score.score(text, model, threshold)

    assert label == False or label == True

# Propensity Value Test: Checking whether the propensity lies in [0,1]
def test_prop_value(text = text, threshold = threshold, model = model) -> None:
    label, prop = score.score(text, model, threshold)

    assert prop >= 0 and prop <= 1

# Checking when threshold is 0 whether the prediction becomes True
def test_pred_thres_0(text = text, threshold = threshold, model = model) -> None:
    label, prop = score.score(text, model, threshold=0)

    assert label == True

# Checking when threshold is 1 whether the prediction becomes False
def test_pred_thres_1(text = text, threshold = threshold, model = model) -> None:
    label, prop = score.score(text, model, threshold=1)

    assert label == False

# Testing obvious spam input
def test_obvious_spam(text = obv_spam_text, threshold = threshold, model = model) -> None:
    label, prop = score.score(text, model, threshold)

    assert label == True

# Testing obvious ham input
def test_obvious_ham(text = obv_ham_text, threshold = threshold, model = model) -> None:
    label, prop = score.score(text, model, threshold)

    assert label == False

def test_flask():
    # Launch the Flask app using os.system
    os.system('start /b python app.py')

    # Wait for the app to start up
    time.sleep(10)

    # Make a request to the endpoint
    response = requests.get('http://127.0.0.1:5000/')
    print(response.status_code)

    # Assert that the response is what we expect
    assert response.status_code == 200

    assert type(response.text) == str

    # Shut down the Flask app using os.system
    os.system('kill $(lsof -t -i:5000)')