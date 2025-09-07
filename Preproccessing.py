import pandas as pd
import re

def llm_safe_clean(text):
    """
    Clean text for LLM embeddings:
    - Lowercase for consistency
    - Remove URLs and HTML tags
    - Remove excessive punctuation/noise
    - Preserve words, grammar, and context
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove excessive punctuation (keep basic punctuation for context if needed)
    text = re.sub(r"[^\w\s.,!?'-]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Load both datasets
true_df = pd.read_csv("Dataset/True.csv")
fake_df = pd.read_csv("Dataset/Fake.csv")

# Add label column
true_df["label"] = 1
fake_df["label"] = 0

# Combine into one DataFrame
df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df = df.drop(columns=["title", "text", "date", "subject"])

df["content"] = df["content"].apply(llm_safe_clean)

df = df.dropna(subset=['content'])
df = df[df['content'].str.strip() != ""].reset_index(drop=True)


# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean dataset

df['content'] = df['content'].astype(str)    # ensure text column is string
df['label'] = df['label'].astype(int)

print(df.info())# make sure labels are int


# Save the combined DataFrame
df.to_csv("Dataset/combined_news.csv", index=False, encoding="utf-8")

print(df['content'].apply(type).value_counts())
print(df.isnull().sum())





