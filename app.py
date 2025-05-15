from flask import Flask, request, render_template, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
# from lime.lime_text import LimeTextExplainer
import spacy
import torch
import numpy as np

app = Flask(__name__)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load sentiment analysis pipeline
MODEL_NAME = 'nlptown/bert-base-multilingual-uncased-sentiment'
sentiment_pipeline = pipeline('sentiment-analysis', model=MODEL_NAME)

# Tokenizer and model for LIME explainability
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# LIME explainer
# class_names = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
# explainer = LimeTextExplainer(class_names=class_names)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        review = data.get('review', '').strip()

        if not review:
            return jsonify({'error': 'No review provided'}), 400

        language = detect(review)
        doc = nlp(review)
        sentences = [sent.text for sent in doc.sents]

        results = []
        for sentence in sentences:
            analysis = sentiment_pipeline(sentence)[0]
            emotion = analysis['label']  # e.g., '4 stars'
            score = round(analysis['score'] * 100, 2)
            results.append({
                'sentence': sentence,
                'emotion': emotion,
                'score': score
            })

        overall_sentiment = max(results, key=lambda x: x['score'])['emotion']

        return jsonify({
            'language': language,
            'overall_sentiment': overall_sentiment,
            'details': results
        })

    return jsonify({'error': 'Invalid content type, JSON expected'}), 400

# @app.route('/explain', methods=['POST'])
# def explain():
#     if request.is_json:
#         try:
#             data = request.get_json()
#             text = data.get('review', '').strip()

#             if not text:
#                 return jsonify({'error': 'No review provided'}), 400

#             def predict_proba(texts):
#                 inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
#                 with torch.no_grad():
#                     outputs = model(**inputs)
#                 probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
#                 return probs

#             exp = explainer.explain_instance(text, predict_proba, num_features=10)
#             explanation = exp.as_list()
#             return jsonify({'explanation': explanation})

#         except Exception as e:
#             import traceback
#             traceback.print_exc()  # logs to terminal
#             return jsonify({'error': str(e)}), 500

#     return jsonify({'error': 'Invalid content type, JSON expected'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)
