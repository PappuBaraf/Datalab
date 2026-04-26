# DAV_Practical/q10_bow.py
# Question 10: Implement Python code to demonstrate Bag of Word [cite: 2]

from sklearn.feature_extraction.text import CountVectorizer

# Sample corpus
corpus = [
    "Data Analytics is fun.",
    "Data Analytics involves data visualization.",
    "Machine learning and data analytics."
]

# Implement Bag of Words
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(corpus)

print("--- Bag of Words (BoW) ---")
print("\nVocabulary (Feature Names):")
print(vectorizer.get_feature_names_out())

print("\nBoW Representation (Matrix):")
print(bow_matrix.toarray())
