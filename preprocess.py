import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# downloading these once, needed for stopwords and lemmatizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # convert to lowercase first
    text = str(text).lower()
    
    # remove anything that's not a letter or space
    text = re.sub(r'[^a-z\s]', '', text)
    
    # split into words
    words = text.split()
    
    # remove stopwords and lemmatize each word
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return ' '.join(cleaned)