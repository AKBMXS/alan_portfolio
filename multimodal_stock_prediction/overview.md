Multimodal Stock Market Prediction System
A research-driven system that predicts next-day stock price movements using a multimodal fusion of candlestick images, technical indicators, and sentiment signals. This project formed the basis of my B.Tech thesis and required designing a complete end-to-end ML pipeline from data collection to model evaluation.

---

Project Overview
The system integrates three distinct modalities:
1. Visual Modality – Candlestick chart images processed using a Vision Transformer (ViT)
2. Tabular Modality – Technical indicators (RSI, MACD, EMA, ATR, OBV, etc.)
3. Sentiment Modality – News & social sentiment embeddings generated using FinBERT

These modalities are fused and passed into a machine learning ensemble (CatBoost) to predict next-day directional movement (up/down).

---

Architecture
1. Data Pipeline
- Automated collection of OHLC data via Yahoo Finance API  
- Generation of candlestick images using Matplotlib  
- Calculation of 20+ technical indicators  
- Extraction of sentiment from StockTwits & Google News using FinBERT  
- 95,000+ samples collected across multiple timeframes  

2. Modeling
- ViT-based embedding extraction  
- Normalized numerical indicators  
- FinBERT sentiment vectorization  
- Modality fusion into a unified feature vector  
- CatBoost classifier for final prediction  

3. Evaluation
- Train/validation/test split  
- Confusion matrix  
- Per-ticker performance analysis  
- Achieved **64% directional accuracy** across diversified equities  

---

Tools & Technologies
- Python  
- Yahoo Finance API  
- Matplotlib  
- TA-Lib / pandas-ta  
- FinBERT  
- HuggingFace Transformers  
- CatBoost  
- NumPy / Pandas  

---

Key Contributions
- Designed the entire multimodal ML architecture  
- Implemented automated data pipelines  
- Engineered indicators & sentiment feature extraction  
- Ran large-scale training experiments across 95K samples  
- Wrote full thesis documentation & evaluation metrics  

---

Outcome
The system demonstrated that combining **visual, numerical, and sentiment modalities** significantly improves predictive performance compared to traditional single-modality approaches.

