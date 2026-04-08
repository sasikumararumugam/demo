import streamlit as st
from transformers import pipeline
import numpy as np
import io
import wave
import PyPDF2

st.title("📄 Text to Audio App")

@st.cache_resource
def load_model():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

tts = load_model()

text = st.text_area("Enter text")

if st.button("Convert to Audio"):
    if not text.strip():
        st.warning("Enter text")
    else:
        try:
            speech = tts(text)

            audio = speech["audio"]
            rate = speech["sampling_rate"]

            #  Ensure numpy array
            audio = np.array(audio)

            #  Normalize safely
            audio = audio / np.max(np.abs(audio))

            # Convert to int16
            audio = (audio * 32767).astype(np.int16)

            # Write proper WAV using wave module
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(rate)
                wf.writeframes(audio.tobytes())

            buffer.seek(0)
            audio_bytes = buffer.read()

            # Debug
            st.write("Audio length:", len(audio_bytes))

            # Play
            st.audio(audio_bytes, format="audio/wav")

            # Download
            st.download_button(
                "Download Audio",
                audio_bytes,
                file_name="output.wav",
                mime="audio/wav"
            )

        except Exception as e:
            st.error(str(e))