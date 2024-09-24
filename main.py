from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load emotion detection model
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

@app.get("/")
def root():
    return {"message": "Emotion detection API is running"}

@app.post("/predict/")
async def predict_emotion(text: str):
    outputs = classifier(text)
    best_prediction = max(outputs[0], key=lambda x: x['score'])
    return {"emotion": best_prediction['label']}
