import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
import pickle

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words("english")
stop_words.extend(stopwords.words("russian"))
stop_words = set(stop_words)


def tokenize_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()

    words = [lemmatizer.lemmatize(word) for word in words]

    return words


def prepare_data(data):
    data['tokens'] = data.Text.apply(tokenize_text)
    data['joined_tokens'] = data.tokens.apply(lambda x: ' '.join(x))

    return data

# train_data = pd.read_pickle('train.csv')
# test_data = pd.read_pickle('test.csv')
#
# test_data = pd.DataFrame(test_data, columns=['Text'])
#
# train_data = prepare_data(train_data)
# test_data = prepare_data(test_data)
#
# X_full = pd.concat([train_data.joined_tokens, test_data.joined_tokens]).reset_index(drop=True)
# assert X_full.shape[0] == train_data.shape[0] + test_data.shape[0]
#
# vectorizer = TfidfVectorizer(max_df=0.55, min_df=5)
# vectorizer.fit(X_full)
#
# X_train_tfidf = vectorizer.transform(train_data.joined_tokens)
# X_test_tfidf = vectorizer.transform(test_data.joined_tokens)
#
# y = train_data.Score
#
# params = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
#           'class_weight': ['balanced', None],
#           'warm_start': [True, False],
#           'early_stopping': [True, False]}
#
# sgd_gs = GridSearchCV(SGDClassifier(), params, n_jobs=-1)
# sgd_gs.fit(X_train_tfidf, y)
#
# with open('model.pkl', 'wb') as file:
#     pickle.dump(sgd_gs.best_estimator_, file)
#
# with open('vectorizer.pkl', 'wb') as file:
#     pickle.dump(vectorizer, file)