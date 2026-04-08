import streamlit as st
from transformers import pipeline
import PyPDF2
import numpy as np
import io
import wave

st.set_page_config(page_title="PDF Voice Assistant", layout="wide")

st.title("📄🔊 PDF Voice Assistant")

# -------------------------------
# Load TTS Model
# -------------------------------
@st.cache_resource
def load_tts():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

tts = load_tts()

# -------------------------------
# Extract PDF Text
# -------------------------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# -------------------------------
# Simple QA (Basic)
# -------------------------------
def get_answer(context, question):
    context = context.lower()
    question = question.lower()

    if question in context:
        return "Yes, this information is present in the document."
    else:
        return "I could not find an exact match, but the document contains related information."

# -------------------------------
# Convert Text → Audio
# -------------------------------
def text_to_audio(text):
    speech = tts(text)

    audio = np.array(speech["audio"])
    rate = speech["sampling_rate"]

    # normalize
    if np.max(np.abs(audio)) != 0:
        audio = audio / np.max(np.abs(audio))

    audio = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())

    buffer.seek(0)
    return buffer.read()

# -------------------------------
# UI
# -------------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully!")

    pdf_text = extract_text(uploaded_file)

    question = st.text_input("Ask a question from the PDF")

    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("Please enter a question")
        else:
            answer = get_answer(pdf_text, question)

            st.subheader("📌 Answer:")
            st.write(answer)

            # Convert to audio
            audio_bytes = text_to_audio(answer)

            st.subheader("🔊 Audio Response:")
            st.audio(audio_bytes, format="audio/wav")

            st.download_button(
                "Download Audio",
                audio_bytes,
                file_name="answer.wav",
                mime="audio/wav"
            )