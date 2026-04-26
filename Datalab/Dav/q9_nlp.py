# DAV_Practical/q9_nlp.py
# Question 9: Implement python program to perform tokenization, stop word removal, stemer and lemitizer in Python. [cite: 2]

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK datasets instantly
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

text = "The quick brown foxes are jumping over the lazy dogs. Data Analytics is amazing!"

# 1. Tokenization
tokens = word_tokenize(text)
print("--- 1. Tokenization ---")
print(tokens)

# 2. Stop Word Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]
print("\n--- 2. Stop Word Removal ---")
print(filtered_tokens)

# 3. Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print("\n--- 3. Stemming ---")
print(stemmed_words)

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\n--- 4. Lemmatization ---")
print(lemmatized_words)