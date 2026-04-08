import streamlit as st
from transformers import pipeline
import scipy.io.wavfile as wavfile
import numpy as np
import io

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

            # Convert float → int16
            audio = (audio * 32767).astype(np.int16)

            # Save to memory
            wav_bytes = io.BytesIO()
            wavfile.write(wav_bytes, rate=rate, data=audio)

            wav_bytes.seek(0)

            # ✅ FIX: use getvalue()
            audio_bytes = wav_bytes.getvalue()

            # Play audio
            st.audio(audio_bytes, format="audio/wav")

            # Download
            st.download_button(
                label="Download Audio",
                data=audio_bytes,
                file_name="output.wav",
                mime="audio/wav"
            )

        except Exception as e:
            st.error(str(e))