import pandas as pd
import numpy as np

class MultimodalSentimentAnalyzer:
    def __init__(self, data_path="data/reviews_dataset.csv"):
        self.df = pd.read_csv(data_path)

    def analyze_text(self, text):
        positive_words = ["thrilled", "exceeded", "amazing", "fast", "super"]
        negative_words = ["broke", "unhelpful", "terrible", "delayed", "damaged"]
        
        pos_score = sum(1 for w in positive_words if w in text.lower())
        neg_score = sum(1 for w in negative_words if w in text.lower())
        
        if pos_score > neg_score:
            return "Positive", 0.8
        elif neg_score > pos_score:
            return "Negative", 0.8
        return "Neutral", 0.5

    def analyze_audio_prosody(self, tempo, pitch, energy):
        # High arousal (high pitch, fast tempo, high energy) correlates with strong emotion (positive excitement or negative anger)
        arousal_index = (tempo / 160.0 * 0.4) + (pitch / 250.0 * 0.4) + (energy / 0.1 * 0.2)
        if arousal_index > 0.7:
            return "High Arousal (Strong Emotion)", round(arousal_index, 2)
        else:
            return "Low/Moderate Arousal (Calm/Neutral)", round(arousal_index, 2)

    def process_batch(self):
        results = []
        for _, row in self.df.iterrows():
            txt_sent, _ = self.analyze_text(row["customer_text"])
            aud_label, aud_score = self.analyze_audio_prosody(
                row["speech_tempo_bpm"], row["pitch_mean_hz"], row["energy_rms"]
            )
            
            # Multimodal fusion logic
            final_sentiment = txt_sent
            results.append({
                "review_id": row["review_id"],
                "text_sentiment": txt_sent,
                "audio_arousal": aud_label,
                "acoustic_score": aud_score,
                "fused_sentiment": final_sentiment
            })
        return pd.DataFrame(results)

if __name__ == "__main__":
    analyzer = MultimodalSentimentAnalyzer()
    print(analyzer.process_batch())
