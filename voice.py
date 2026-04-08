import streamlit as st
from transformers import pipeline
import scipy.io.wavfile as wavfile
import numpy as np
import tempfile

st.title("📄 Text → Audio App (Hugging Face)")

st.write("Convert text or text file into speech")

# -------------------------------
# Load TTS Model
# -------------------------------
@st.cache_resource
def load_tts():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

tts_model = load_tts()

# -------------------------------
# Text Input
# -------------------------------
text = st.text_area("✍️ Enter your text:")

# -------------------------------
# File Upload
# -------------------------------
file = st.file_uploader("📂 Upload .txt file", type=["txt"])

if file:
    text = file.read().decode("utf-8")
    st.subheader("📄 File Content")
    st.write(text)

# -------------------------------
# Convert Button
# -------------------------------
if st.button("🔊 Convert to Audio"):
    if text.strip() == "":
        st.warning("⚠️ Please enter text or upload a file")
    else:
        st.info("🔄 Generating audio...")

        try:
            # 🔥 Handle long text (split into chunks)
            chunks = [text[i:i+200] for i in range(0, len(text), 200)]

            full_audio = []
            sampling_rate = None

            for chunk in chunks:
                speech = tts_model(chunk)

                audio = speech["audio"]
                sampling_rate = speech["sampling_rate"]

                # ✅ FIX: Convert float32 → int16
                audio = (audio * 32767).astype(np.int16)

                full_audio.append(audio)

            # Combine all chunks
            final_audio = np.concatenate(full_audio)

            # Save audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                wavfile.write(tmpfile.name, rate=sampling_rate, data=final_audio)
                audio_path = tmpfile.name

            st.success("✅ Audio generated successfully!")
            st.audio(audio_path)

            # Optional download
            with open(audio_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Audio",
                    data=f,
                    file_name="output.wav",
                    mime="audio/wav"
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")