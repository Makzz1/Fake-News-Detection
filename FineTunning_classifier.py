import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

# 1. Load dataset
df = pd.read_csv("Dataset/combined_news.csv")  # has 'content' and 'label'
dataset = Dataset.from_pandas(df)

# 2. Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["content"], padding="max_length", truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)

# 3. Train-test split
dataset = dataset.train_test_split(test_size=0.2)

# 4. Load model (classification head)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. Training args
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
)

# 6. Trainer
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 7. Train
trainer.train()

# Save fine-tuned model
trainer.save_model("./fine_tuned_classifier")
