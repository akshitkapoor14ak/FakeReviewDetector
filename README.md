# FakeReview Detector 🔍

Ever wondered if that 5-star review on Flipkart is even real? This project tries to answer that.

I built a machine learning model that reads a product review and tells you whether it's genuine or AI-generated/fake — with a confidence score.

Trained on 40,000+ labeled reviews from Kaggle.

---

## What it can do

- paste any review and get an instant fake/genuine verdict
- shows how confident the model is (in %)
- bulk check reviews by uploading a CSV file
- works on any product review — not just one category

---

## Stack used

- Python
- Pandas
- NLTK — for cleaning the text
- Scikit-learn — TF-IDF + Logistic Regression
- Joblib — saving and loading the model
- Streamlit — for the web interface

---

## How the model works

pretty straightforward pipeline:

1. clean the review text (lowercase, remove stopwords, lemmatize)
2. convert cleaned text to numbers using TF-IDF
3. pass it through Logistic Regression
4. get prediction → Fake (CG) or Genuine (OR) + confidence %

ended up with **87% accuracy** on a test set of 8000+ reviews which i think is decent for a first attempt

---

## Folder structure

```
FakeReviewDetector/
├── data/
│   └── reviews.csv
├── model/
│   └── fake_review_model.pkl
│   └── tfidf_vectorizer.pkl
├── app.py
├── model.py
├── preprocess.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## Run it locally

```bash
git clone https://github.com/akshitkapoor14ak/FakeReviewDetector.git
cd FakeReviewDetector
pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

note: run train_model.py first — it generates the .pkl files that the app needs

---

## Dataset

used the Fake Reviews Dataset from Kaggle — around 40k reviews labeled as CG (computer generated) or OR (original). link in references.

---

*Akshit Kapoor — B.Tech AI & Data Science, Delhi NCR*

---

*Akshit Kapoor — B.Tech AI & Data Science, Delhi NCR*
