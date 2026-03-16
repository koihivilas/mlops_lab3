import pandas as pd
import re
import os
import shutil
import joblib
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required nltk resources quietly
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# 1. Text Cleaning Function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple lemmatization
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized)

print("Loading data from Volumes...")
df = pd.read_csv("/Volumes/workspace/default/data/dataset.csv")

df['combined_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
df['cleaned_text'] = df['combined_text'].apply(clean_text)

df = df[df['cleaned_text'].str.len() > 10]
df = df.dropna(subset=['queue'])

print(f"Data prepared: {len(df)} rows.")

X = df['cleaned_text'].values
y = df['queue'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Fitting TF-IDF Model (with enhanced parameters)...")
vectorizer = TfidfVectorizer(
    max_features=8000,      
    ngram_range=(1, 2),     
    min_df=5,               # Remove words appearing in less than 5 tickets
    max_df=0.85,            # Remove words appearing in >85% of texts (templates, greetings)
    sublinear_tf=True       # Apply sublinear tf scaling
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

class_names = list(sorted(set(y)))

# Save preprocessed components to local disk first, then copy to Volumes
# Databricks Volumes do not support random writes (required by .npz/.joblib)
os.makedirs("/tmp/features", exist_ok=True)
os.makedirs("/Volumes/workspace/default/data/features", exist_ok=True)

tmp_paths = {
    "X_train_tfidf.npz": lambda p: scipy.sparse.save_npz(p, X_train_tfidf),
    "X_test_tfidf.npz": lambda p: scipy.sparse.save_npz(p, X_test_tfidf),
    "y_train.joblib": lambda p: joblib.dump(y_train, p),
    "y_test.joblib": lambda p: joblib.dump(y_test, p),
    "vectorizer.joblib": lambda p: joblib.dump(vectorizer, p),
    "classes.joblib": lambda p: joblib.dump(class_names, p)
}

for filename, save_func in tmp_paths.items():
    local_path = f"/tmp/features/{filename}"
    volume_path = f"/Volumes/workspace/default/data/features/{filename}"
    
    # 1. Save locally
    save_func(local_path)
    # 2. Copy to Volume
    shutil.copyfile(local_path, volume_path)

print("Data prep and feature engineering complete. Artifacts saved to /Volumes/workspace/default/data/features/")
