from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model_filename = 'saved_model'
loaded_model = tf.saved_model.load(model_filename)

import pickle
with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_result = None
    if request.method == 'POST':
        headline = request.form['headline']
        sentiment_result = predict_sentiment(headline)
    return render_template('index.html', result=sentiment_result)

def predict_sentiment(headline):
    
    X = loaded_vectorizer.transform([headline]).toarray()

    prediction = loaded_model(X)

    sentiment_labels = ["neutral", "positive", "negative"]
    predicted_sentiment = sentiment_labels[np.argmax(prediction)]
    return predicted_sentiment

if __name__ == '__main__':
    app.run(debug=True)

