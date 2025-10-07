# train_model.py
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data
X = [
    "I love this product",
    "This is amazing",
    "I hate this",
    "This is terrible",
    "Not good",
    "Very happy with it",
    "I dislike this item"
]
y = ["positive", "positive", "negative", "negative", "negative", "positive", "negative"]

# Convert text to numerical features
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)

# Train the model
model = MultinomialNB()
model.fit(X_vect, y)

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")
