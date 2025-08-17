from flask import Flask, render_template, request
from preprocessing import normalize_text
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer using absolute paths
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text = normalize_text(text)
    features = vectorizer.transform([text])
    result = model.predict(features)
    return render_template('index.html', result=result[0])
app.run(debug=True)