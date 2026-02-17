import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

app = FastAPI(title="Fake News Detector API", version="2.0.0")

# --- 1. CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CONTRASTIVE_PATH = MODELS_DIR / "my_contrastive_model"
CLASSIFIER_PATH = MODELS_DIR / "final_news_classifier_pack"
WEIGHTS_PATH = CLASSIFIER_PATH / "attention_model_state.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. DEFINE THE MODEL ARCHITECTURE ---
# We must define the class structure exactly as we did during training
class AttentionClassifier(nn.Module):
    def __init__(self, model_path, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size
        self.attention_dense = nn.Linear(hidden_size, hidden_size)
        self.attention_vector = nn.Linear(hidden_size, 1, bias=False)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state
        u = torch.tanh(self.attention_dense(H))
        scores = self.attention_vector(u).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        alpha = F.softmax(scores, dim=1)
        context_vector = torch.sum(H * alpha.unsqueeze(-1), dim=1)
        logits = self.classifier(context_vector)
        return logits, alpha


# --- 3. GLOBAL STATE ---
model_state = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


@app.on_event("startup")
def load_models():
    print("Loading models...")

    # A. Load Tokenizer (from the classifier pack)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(CLASSIFIER_PATH))
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise e

    # B. Initialize Model Structure (using the backbone config)
    # We point to 'my_contrastive_model' to get the BERT structure
    model = AttentionClassifier(str(CONTRASTIVE_PATH))

    # C. Load the Trained Weights
    # We load the specific attention weights we saved
    try:
        state_dict = torch.load(str(WEIGHTS_PATH), map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise e

    model.to(device)
    model.eval()  # Set to evaluation mode (freezes dropout, etc.)

    # Save to global state
    model_state["model"] = model
    model_state["tokenizer"] = tokenizer
    print(f"Model loaded successfully on {device}")


@app.get("/health")
def health():
    return {"status": "ok", "model": "Attention-Based-v1"}


@app.post("/predict")
def predict(req: PredictRequest):
    model = model_state["model"]
    tokenizer = model_state["tokenizer"]

    # 1. Tokenize
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2. Inference
    with torch.no_grad():
        logits, alphas = model(input_ids, attention_mask)

        # Get Prediction
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

        # Get Attention Scores
        scores = alphas[0].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # 3. Format "Highlights" with reconstruction
    highlights = []
    current_word = ""
    current_score = 0.0
    subtoken_count = 0

    for token, score in zip(tokens, scores):
        # Skip special tokens
        if token in ["[PAD]", "[CLS]", "[SEP]"]:
            continue

        # Check if it's a subword (starts with ##)
        if token.startswith("##"):
            # Append to the previous word
            current_word += token.replace("##", "")
            # Average the score
            current_score = (current_score * subtoken_count + score) / (subtoken_count + 1)
            subtoken_count += 1
        else:
            # If we have a previous word stored, save it now
            if current_word != "":
                highlights.append({"token": current_word, "score": float(current_score)})

            # Start a new word
            current_word = token
            current_score = score
            subtoken_count = 1

    # If there's a remaining word after the loop, add it
    if current_word != "":
        highlights.append({"token": current_word, "score": float(current_score)})

    # 4. Gemini Fact Check
    gemini_status = "no"  # Default if key is missing or error
    gemini_reason = "Could not verify."
    try:
        import google.generativeai as genai
        import json
        
        # --- USER: REPLACE THIS WITH YOUR ACTUAL API KEY ---
        GEMINI_API_KEY = "AIzaSyApfzVcknoE5JVpGaNpTrJ_WQwO7eUC4A8" 
        
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            # Use 'gemini-2.5-flash' as verified from available models list
            model_gemini = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = f"""
            Analyze the following text for factual accuracy.
            Provide your response in raw JSON format with two keys:
            1. "status": One of "true" (fact), "false" (fake/falsehood), or "no" (opinion/vague).
            2. "reason": A single short sentence explaining why.
            
            Text: {req.text}
            """
            
            response = model_gemini.generate_content(prompt)
            # Clean up potential markdown formatting in response
            clean_response = response.text.strip().replace("```json", "").replace("```", "")
            result_json = json.loads(clean_response)
            
            gemini_status = result_json.get("status", "no").lower()
            gemini_reason = result_json.get("reason", "No reason provided.")
            
    except Exception as e:
        print(f"Gemini API Error: {e}")
        gemini_status = "error"
        gemini_reason = f"Error: {str(e)}"

    return {
        "is_fake": bool(pred_label == 0),
        "confidence": confidence,
        "attention_data": highlights,
        "gemini_verification": gemini_status,
        "gemini_reason": gemini_reason
    }
