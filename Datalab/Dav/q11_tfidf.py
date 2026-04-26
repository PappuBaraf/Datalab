# DAV_Practical/q11_tfidf.py
# Question 11: Implement Python code to demonstrate TFIDF. [cite: 2]

from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus
corpus = [
    "The sky is blue.",
    "The sun is bright today.",
    "The sun in the sky is bright."
]

# Implement TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

print("--- Term Frequency-Inverse Document Frequency (TF-IDF) ---")
print("\nVocabulary (Feature Names):")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Representation (Matrix):")
# Rounding values for better readability
import numpy as np
print(np.round(tfidf_matrix.toarray(), 3))
