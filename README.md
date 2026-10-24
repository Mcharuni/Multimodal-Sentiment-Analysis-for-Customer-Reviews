# Multimodal Sentiment Analysis for Customer Reviews

A machine learning pipeline that combines text review sentiment analysis with acoustic prosody features (pitch, energy, and speech tempo) to capture true customer emotion, sarcasm, and satisfaction.

## Features
- **Text Sentiment Classifier:** Evaluates lexical polarity from customer feedback.
- **Acoustic Feature Extractor:** Analyzes vocal tone, tempo, and energy using audio metrics.
- **Multimodal Fusion Engine:** Integrates both text and audio signals for improved classification accuracy.
- **FastAPI REST Service:** Ready for real-time app deployment.

## Project Structure
```text
multimodal_sentiment_analysis/
├── data/
│   └── reviews_dataset.csv
├── src/
│   ├── multimodal_analyzer.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run analysis script: `python src/multimodal_analyzer.py`
3. Start API server: `uvicorn src.app:app --reload`
