from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
from sentence_transformers import SentenceTransformer
from pathlib import Path


app = FastAPI(title="Fake News Detector API", version="1.0.0")


# CORS (adjust origins for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

class PredictRequest(BaseModel):
    text: str

class PredictBatchRequest(BaseModel):
    texts: List[str]


@app.on_event("startup")
def load_models():
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    # Load classifier
    app.state.clf = joblib.load(str(models_dir / "logistic_model.joblib"))

    # Load embedding model
    app.state.embed_model = SentenceTransformer(str(models_dir / "sentence_transformer_model"))
    print("Models loaded")

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.post("/predict")
def predict(req: PredictRequest):
    embedding = app.state.embed_model.encode([req.text])
    pred = app.state.clf.predict(embedding)[0]
    return {"Result": bool(pred)}

