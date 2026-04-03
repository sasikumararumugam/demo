import streamlit as st
from transformers import pipeline

# Title
st.title("💬 Sentiment Analysis App")

st.write("Analyze text sentiment using Hugging Face Transformers")

# Load model (cache for performance)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

# User input
user_input = st.text_area("Enter your text here:")

# Button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result = classifier(user_input)

        sentiment = result[0]["label"]
        score = result[0]["score"]

        st.subheader("Result")

        # Display nicely
        if sentiment == "POSITIVE":
            st.success(f"😊 Positive (Confidence: {score:.2f})")
        else:
            st.error(f"😠 Negative (Confidence: {score:.2f})")