import streamlit as st
from transformers import pipeline
import scipy.io.wavfile as wavfile
import numpy as np

st.title("📄 Text to Audio App")

# Load model
@st.cache_resource
def load_model():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

tts = load_model()

# Input
text = st.text_area("Enter text")

# Convert
if st.button("Convert to Audio"):
    if text.strip() == "":
        st.warning("Enter text")
    else:
        try:
            speech = tts(text)

            audio = speech["audio"]
            rate = speech["sampling_rate"]

            # ✅ FIX: float → int16
            audio = (audio * 32767).astype(np.int16)

            wavfile.write("output.wav", rate=rate, data=audio)

            st.audio("output.wav")

        except Exception as e:
            st.error(str(e))