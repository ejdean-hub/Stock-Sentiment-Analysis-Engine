# Stock-Sentiment-Analysis-Engine


A machine‑learning pipeline that fetches real‑time news from Alpha Vantage, cleans and processes the text, trains a sentiment classifier, visualizes results, and saves a reusable model for future predictions.

**Overview**

This project builds a 3‑class sentiment analysis model (Positive, Neutral, Negative) using:

-Alpha Vantage News Sentiment API

-TF‑IDF text vectorization

-Linear Support Vector Machine (SVM)

-Matplotlib & Seaborn visualizations

-Joblib model serialization

-The engine automatically:

-Fetches news for selected stock tickers

-Cleans and preprocesses the text

-Collapses Alpha Vantage’s 5 sentiment labels → 3 classes

-Splits data chronologically (80/20)

-Trains a Linear SVM classifier

-Evaluates performance

-Generates visualizations

-Saves the trained model (stock_sentiment.pkl)

**Potential Future Upgrades:**

-Deploy as an API endpoint

-Add real‑time streaming sentiment

-Integrate with trading signals

-Add dashboard (Streamlit / Dash)
