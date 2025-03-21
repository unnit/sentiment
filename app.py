from flask import Flask, request, render_template, jsonify
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
    if request.is_json:
        data = request.get_json()
        review = data.get('review', '')

        # Validate input
        if not review:
            return jsonify({'error': 'No review provided'}), 400

        # Get prediction from model
        prediction = learn.predict(review)
        sentiment = 'Positive' if prediction[0] == 'pos' else 'Negative'
        probability = round(prediction[2][1].item(), 4)

        # Return JSON response
        return jsonify({
            'sentiment': sentiment,
            'confidence': f'{probability * 100:.2f}%'
        })

    # If not JSON, return an error
    return jsonify({'error': 'Invalid content type, JSON expected'}), 400

if __name__ == '__main__':
    app.run(debug=True)
