import joblib
from preprocess import clean_text

# loading the saved model and vectorizer
model = joblib.load('model/fake_review_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

def predict_review(review_text):
    # clean the input text first
    cleaned = clean_text(review_text)
    
    # convert to tfidf numbers
    vectorized = tfidf.transform([cleaned])
    
    # get prediction and confidence
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized)[0].max()
    
    return prediction, round(confidence * 100, 2)