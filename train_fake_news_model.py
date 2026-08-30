"""
Train the fake-news text classifier used by pages/Fakenews.py.

Expected input: one or two CSVs with columns ['text', 'label'] where
label = 1 for REAL, 0 for FAKE (matches the convention in Fakenews.py).

If you download the Kaggle "Fake and Real News Dataset", it comes as
Fake.csv and True.csv (title, text, subject, date) with no label column —
this script merges and labels them automatically if it finds that layout.

Usage:
    python train_fake_news_model.py --fake Fake.csv --real True.csv
    # or, if you already have a single labeled csv:
    python train_fake_news_model.py --data news.csv
"""

import argparse
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_kaggle_style(fake_path, real_path):
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    fake["label"] = 0  # FAKE
    real["label"] = 1  # REAL
    df = pd.concat([fake, real], ignore_index=True)
    # Kaggle set has 'title' + 'text'; combine for richer signal
    if "title" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    return df[["text", "label"]]


def load_single_csv(path):
    df = pd.read_csv(path)
    assert {"text", "label"}.issubset(df.columns), \
        "CSV must have 'text' and 'label' columns (label: 1=REAL, 0=FAKE)"
    return df[["text", "label"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", help="Path to Fake.csv (Kaggle-style)")
    parser.add_argument("--real", help="Path to True.csv (Kaggle-style)")
    parser.add_argument("--data", help="Path to a single labeled CSV")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    if args.data:
        df = load_single_csv(args.data)
    elif args.fake and args.real:
        df = load_kaggle_style(args.fake, args.real)
    else:
        raise SystemExit("Provide either --data OR both --fake and --real")

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size,
        random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english", max_df=0.7, ngram_range=(1, 2), max_features=50000
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # NOTE: Fakenews.py calls model.predict_proba(), so the model must support
    # probability outputs. LogisticRegression does this natively.
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}\n")
    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"]))

    joblib.dump(model, "fake_news_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("\nSaved fake_news_model.pkl and vectorizer.pkl")


if __name__ == "__main__":
    main()
