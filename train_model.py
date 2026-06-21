import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from preprocess import clean_text

# loading the dataset
df = pd.read_csv('data/reviews.csv')

# keeping only the columns we need
df = df[['text_', 'label']]

# dropping empty rows if any
df.dropna(inplace=True)

# cleaning the review text
print("cleaning reviews... this might take a minute")
df['cleaned'] = df['text_'].apply(clean_text)

# splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label'], test_size=0.2, random_state=42
)

# converting text to numbers using tfidf
tfidf = TfidfVectorizer(max_features=5000)
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

# training logistic regression model
print("training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tf, y_train)

# checking accuracy
y_pred = model.predict(X_test_tf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# saving model and vectorizer
joblib.dump(model, 'model/fake_review_model.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')
print("model saved!")