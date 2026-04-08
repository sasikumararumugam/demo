import streamlit as st
from transformers import pipeline
import scipy.io.wavfile as wavfile

st.title("📄 Text → Audio App")

# Load TTS model
tts_model = pipeline("text-to-speech", model="facebook/mms-tts-eng")

# Text input
text = st.text_area("Enter your text:")

# Text file upload
file = st.file_uploader("Or upload text file", type=["txt"])

# If file uploaded, read text
if file:
    text = file.read().decode("utf-8")
    st.write("📄 File Content:")
    st.write(text)

# Convert to audio
if st.button("🔊 Convert to Audio"):
    if text.strip() == "":
        st.warning("Please enter text or upload file")
    else:
        st.info("🔄 Generating audio...")

        speech = tts_model(text)

        wavfile.write(
            "output.wav",
            rate=speech["sampling_rate"],
            data=speech["audio"]
        )

        st.audio("output.wav")