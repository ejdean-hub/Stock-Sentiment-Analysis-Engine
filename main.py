import os
from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

import matplotlib.pyplot as plt
import seaborn as sns




# CONFIG
TICKERS = [
    "NVDA",
    "GOOG",
    "AAPL",
    "MSFT",
    "IBM"
]

TOPICS = "technology"
LIMIT = 1000



# FETCH NEWS FOR ONE TICKER
def fetch_news_sentiment(ticker, topics, time_from=None, time_to=None):
    """
    Fetches news sentiment for a given ticker and topics within a time range from Alpha Vantage.

    Parameters:
    :param ticker: stock ticker
    :param topics: article topics
    :param time_from:
    :param time_to:
    :return: data frame of news sentiment
    """
    load_dotenv()
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "topics": topics,
        "limit": LIMIT,
        "apikey": API_KEY
    }

    if time_from:
        params["time_from"] = time_from
    if time_to:
        params["time_to"] = time_to

    response = requests.get(base_url, params=params)
    data = response.json()

    if "feed" not in data: # verifies whether news exists for ticker
        print(f"No feed returned for {ticker}. Full response:")
        print(data)
        return pd.DataFrame()

    df = pd.json_normalize(data["feed"])
    df["ticker"] = ticker
    return df



# COLLAPSE SENTIMENT LABELS (5 → 3 classes)
def collapse_sentiment(label):
    """
    There are originally 5 sentiment labels from Alpha Vantage (positive, somewhat positive, neutral, somewhat negative, and negative).
    This collapses them to positive, neutral, and negative. Bearish => negative, Bullish => positive (used interchangeably).

    :param label: Sentiment label
    :return: filtered labels
    """
    if "Bearish" in label:
        return "Negative"
    elif "Bullish" in label:
        return "Positive"
    else:
        return "Neutral"


# CLEAN TEXT
def prepare_text(df):
    """
     Processes text to make it readable for the model. Creates a new text column by combining title and summary.

    :param df:
    :return: cleaned copy of df
    """
    if df.empty:
        print("No data to clean — empty DataFrame.")
        return df

    df = df.copy()

    df["text"] = (
        df["title"].fillna("") + " " +
        df["summary"].fillna("")
    )

    df["text"] = (
        df["text"]
        .str.replace(r"[^a-zA-Z]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
        .str.strip()
    )

    return df


def main():
    # ---------------------------------------------------------
    # MAIN INGESTION
    # ---------------------------------------------------------
    print("Fetching news...")

    time_from = (datetime.utcnow() - timedelta(days=30)).strftime("%Y%m%dT%H%M")

    all_dfs = []
    for ticker in TICKERS:
        print(f"Fetching news for {ticker}...")
        df_ticker = fetch_news_sentiment(ticker, TOPICS, time_from=time_from)
        if not df_ticker.empty:
            all_dfs.append(df_ticker)

    if not all_dfs:
        print("No articles returned for ANY ticker. Exiting.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    # ---------------------------------------------------------
    # CLEAN + LABELS
    # ---------------------------------------------------------
    df_clean = prepare_text(df)
    df_clean["sentiment_3"] = df_clean["overall_sentiment_label"].apply(collapse_sentiment)

    print(df_clean.head())
    print("Total articles:", len(df_clean))

    # ---------------------------------------------------------
    # SORT BY TIME
    # ---------------------------------------------------------
    df_clean["time_published"] = pd.to_datetime(df_clean["time_published"])
    df_clean = df_clean.sort_values("time_published")

    # ---------------------------------------------------------
    # TRAIN/TEST SPLIT (80/20 by time)
    # ---------------------------------------------------------
    split_point = df_clean["time_published"].quantile(0.8)
    train_df = df_clean[df_clean["time_published"] < split_point]
    test_df = df_clean[df_clean["time_published"] >= split_point]

    train_texts = train_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    y_train = train_df["sentiment_3"]
    y_test = test_df["sentiment_3"]

    # ---------------------------------------------------------
    # Term Frequency–Inverse Document Frequency (TF-IDF) VECTORIZER
    # ---------------------------------------------------------
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), # captures one word and two word phrases, for context
        min_df=2,           # ignores words that appear in less than two documents
        max_df=0.9,         # ignores words that appear in 90% of documents
        sublinear_tf=True   # applies log scaling.
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # ---------------------------------------------------------
    # MODEL USED: LINEAR SVM
    # ---------------------------------------------------------
    clf = LinearSVC()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(confusion_matrix(y_test, preds))

    # ---------------------------------------------------------
    # VISUALIZATIONS
    # ---------------------------------------------------------

    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d",
                cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # 2. Sentiment Distribution
    plt.figure(figsize=(6, 4))
    df_clean["sentiment_3"].value_counts().plot(kind="bar",
        color=["red", "gray", "green"])
    plt.title("Sentiment Distribution (3‑Class)")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # 3. Sentiment Over Time
    df_daily = df_clean.groupby([df_clean["time_published"].dt.date,
                                 "sentiment_3"]).size().unstack(fill_value=0)

    df_daily.plot(kind="line", figsize=(10, 5))
    plt.title("Sentiment Over Time")
    plt.xlabel("Date")
    plt.ylabel("Article Count")
    plt.tight_layout()
    plt.show()

    # 4. Per‑Ticker Sentiment Distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_clean, x="ticker", hue="sentiment_3",
                  palette={"Negative": "red", "Neutral": "gray", "Positive": "green"})
    plt.title("Sentiment by Ticker")
    plt.xlabel("Ticker")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------
    joblib.dump((vectorizer, clf), "stock_sentiment.pkl")
    print("Model saved as stock_sentiment.pkl")


if __name__ == "__main__":
    main()


