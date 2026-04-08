import streamlit as st
from transformers import pipeline
import scipy.io.wavfile as wavfile
import numpy as np
import io

st.title("📄 Text to Audio App")

st.write("Convert text into speech using Hugging Face")

# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

tts = load_model()

# -------------------------------
# Input
# -------------------------------
text = st.text_area("✍️ Enter your text:")

# -------------------------------
# Convert Button
# -------------------------------
if st.button("🔊 Convert to Audio"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        try:
            st.info("🔄 Generating audio...")

            # Generate speech
            speech = tts(text)

            audio = speech["audio"]
            rate = speech["sampling_rate"]

            # ✅ Convert float32 → int16
            audio = (audio * 32767).astype(np.int16)

            # ✅ Save to memory instead of file
            wav_bytes = io.BytesIO()
            wavfile.write(wav_bytes, rate=rate, data=audio)

            st.success("✅ Audio generated successfully!")

            # 🔊 Play audio
            st.audio(wav_bytes.getvalue(), format="audio/wav")

            # ⬇️ Download option
            st.download_button(
                label="⬇️ Download Audio",
                data=wav_bytes.getvalue(),
                file_name="output.wav",
                mime="audio/wav"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")