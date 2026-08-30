import os
import streamlit as st
import joblib
import re
import requests
from googletrans import Translator

# === Load model and vectorizer ===
# NOTE: Ensure these files exist in the same directory as your script.
try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    st.error("Error: Model or vectorizer files not found. Please ensure 'fake_news_model.pkl' and 'vectorizer.pkl' are in the same directory.")
    st.stop()

# === Translate non-English text to English ===
def translate_text(text):
    translator = Translator()
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        st.warning(f"Translation failed, using original text. Error: {e}")
        return text

# === Predict fake news using the loaded model ===
# This is the function that was missing!
def predict_fake_news(text):
    # Preprocess the text using the loaded vectorizer
    vectorized_text = vectorizer.transform([text])
    
    # Get the prediction and the confidence score (probability)
    prediction = model.predict(vectorized_text)[0]
    confidence_scores = model.predict_proba(vectorized_text)[0]
    
    # Get the confidence for the predicted class
    confidence = confidence_scores[prediction]
    
    return prediction, confidence

# === Check GNews for presence of article ===
def check_with_gnews(query):
    # Set your key as an environment variable: export GNEWS_API_KEY=your_key
    api_key = os.environ.get("GNEWS_API_KEY")
    if not api_key:
        return False, None
    url = f"https://gnews.io/api/v4/search?q={query}&token={api_key}&lang=en"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("articles"):
            return True, data["articles"][0]["title"]
        else:
            return False, None
    except:
        return False, None

# === Scam/phishing pattern detection ===
def detect_scam_patterns(text):
    scam_keywords = ['free', 'cashback', 'click here', 'win', 'modi govt', 'offer', 'gift', 'scam']
    for keyword in scam_keywords:
        if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
            return True
    return False

# === Streamlit UI ===
st.set_page_config(page_title="FAKEBUSTER - Fake News Detector", page_icon="📰")
st.title("📰 FAKEBUSTER - Fake News Detection System")
st.write("Enter a news headline or short article below:")

input_text = st.text_area("News Article/Headline", height=150)

if st.button("Analyze"):
    if not input_text.strip():
        st.warning("Please enter some news text to analyze.")
    else:
        st.subheader("Analysis Result")

        # Step 1: Translation
        with st.spinner("Translating..."):
            translated = translate_text(input_text)
            st.markdown(f"**Translated to English:** `{translated}`")

        # Step 2: ML Prediction
        with st.spinner("Analyzing with ML model..."):
            prediction, confidence = predict_fake_news(translated)
            label = "🟢 REAL" if prediction == 1 else "🔴 FAKE"
            st.markdown(f"**ML Prediction:** `{label}` (Confidence: {confidence:.2f})")

        # Step 3: GNews API Check
        with st.spinner("Searching GNews..."):
            found, article_title = check_with_gnews(translated)
            if found:
                st.success(f"✅ Found in recent articles: {article_title}")
            else:
                st.warning("No recent article found via GNews.")

        # Step 4: Scam/Phishing Pattern
        is_scam = detect_scam_patterns(translated)
        if is_scam:
            st.error("Scam Pattern Detected: This resembles a phishing or promotional scam message.")
        else:
            st.info("✅ No scam/phishing pattern detected.")

        # Step 5: Final Verdict Logic
        st.subheader("Final Verdict")

        if is_scam:
            st.error("🔴 This news is **FAKE** due to scam/phishing patterns.")
        elif prediction == 0 and confidence >= 0.7:
            st.error("🔴 This news is **FAKE** as predicted by the ML model with high confidence.")
        elif prediction == 1 and confidence >= 0.6:
            st.success("🟢 This news is **REAL** and credible as per the ML model.")
        else:
            st.warning("🟡 This news could not be verified confidently. Please fact-check from multiple sources.")

# Footer
st.markdown("---")

