import joblib
from sentence_transformers import SentenceTransformer

# Load classifier
clf = joblib.load("app/models/logistic_model.joblib")

# Load embedding model
embed_model = SentenceTransformer("app/models/sentence_transformer_model")

text = "The government has announced a new policy for education."  # example news text

embedding = embed_model.encode([text])  # returns a list of embeddings

pred = clf.predict(embedding)[0]  # 0 or 1
print("Prediction:", pred)
print("Fake News?" , "Yes" if pred == 0 else "No")
