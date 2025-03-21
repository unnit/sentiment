from flask import Flask, request, render_template
import pickle
import torch
from fastai.text.all import *

app = Flask(__name__)

# Load the pre-trained AWD_LSTM model
model_path = 'model/awdlstm_text_classifier.pkl'
learn = load_learner(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    prediction = learn.predict(review)
    
    # Extract sentiment (e.g., 'neg' or 'pos') from the FastAI output
    sentiment = 'Positive' if prediction[0] == 'pos' else 'Negative'
    probability = round(prediction[2][1].item(), 4)  # Probability of positive sentiment
    
    return render_template('index.html', prediction_text=f'Sentiment: {sentiment} (Confidence: {probability * 100:.2f}%)')

if __name__ == '__main__':
    app.run(debug=True)

