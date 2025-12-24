from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(texts):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
