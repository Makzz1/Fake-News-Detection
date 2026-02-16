from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import pandas as pd

# 1. Load dataset
df = pd.read_csv("../Dataset/combined_news.csv")

# 2. Prepare Data for Contrastive Learning
# SBERT needs data in a specific 'InputExample' format
# We create a list where each item is the text and its label (0 or 1)
train_examples = []
for i, row in df.iterrows():
    train_examples.append(InputExample(texts=[row['content']], label=int(row['label'])))

# 3. Create a DataLoader
# This batches the data so the model can compare multiple articles at once
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# 4. Load the Model (The "Brain" we are going to train)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5. Define the Contrastive Loss
# BatchHardTripletLoss forces the model to separate the classes in the vector space
# It requires the embeddings to be useful for clustering
train_loss = losses.BatchHardTripletLoss(model=model)

# 6. Run Contrastive Fine-Tuning
# This is where the "reshaping" of the embedding space happens
print("Starting Contrastive Fine-Tuning...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,  # 1 or 2 epochs is usually enough for contrastive tuning
    show_progress_bar=True,
    output_path="./fine_tuned_sbert_contrastive"
)

print("Contrastive Learning Complete. Model saved to './fine_tuned_sbert_contrastive'")