from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("../Dataset/combined_news.csv")

# 2. Load pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Encode content
embeddings = model.encode(df["content"].tolist(), batch_size=32, show_progress_bar=True)

# 4. Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(embeddings, df["label"], test_size=0.2, random_state=42)

# 5. Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
