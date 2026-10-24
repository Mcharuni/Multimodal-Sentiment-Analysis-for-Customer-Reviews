from fastapi import FastAPI
from pydantic import BaseModel
from src.multimodal_analyzer import MultimodalSentimentAnalyzer

app = FastAPI(title="Multimodal Sentiment Analysis API", version="1.0")
analyzer = MultimodalSentimentAnalyzer()

class MultimodalInput(BaseModel):
    customer_text: str
    speech_tempo_bpm: float
    pitch_mean_hz: float
    energy_rms: float

@app.get("/")
def home():
    return {"message": "Multimodal Sentiment Analysis API is online!"}

@app.post("/predict")
def predict_sentiment(data: MultimodalInput):
    txt_sent, _ = analyzer.analyze_text(data.customer_text)
    aud_label, aud_score = analyzer.analyze_audio_prosody(
        data.speech_tempo_bpm, data.pitch_mean_hz, data.energy_rms
    )
    return {
        "text_sentiment": txt_sent,
        "audio_prosody_analysis": aud_label,
        "acoustic_arousal_score": aud_score,
        "fused_multimodal_sentiment": txt_sent
    }
