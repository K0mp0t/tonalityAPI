import pandas as pd
from flask import Flask, request
import pickle
from prediction_module import prepare_data
import nltk
from nltk.corpus import stopwords
import json

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words("english")+stopwords.words("russian"))


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = json.loads(request.data)
        data = pd.DataFrame(data, columns=['Text'])
        data = prepare_data(data)
        vectorized_data = vectorizer.transform(data.joined_tokens)

        data['prediction'] = model.predict(vectorized_data)

        return data[['Text', 'prediction']].to_json()


if __name__ == '__main__':
    app.run()
