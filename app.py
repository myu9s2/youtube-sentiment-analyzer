import streamlit as st
import yt_dlp
import whisper
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tempfile
import os

# ==================================
# CONFIG
# ==================================

st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    layout="wide"
)

PATH_ROBERTA = "models/roberta_sentiment"

# ==================================
# LOAD MODELS
# ==================================

@st.cache_resource
def load_models():

    tokenizer = AutoTokenizer.from_pretrained(
        PATH_ROBERTA
    )

    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        PATH_ROBERTA
    )

    whisper_model = whisper.load_model(
        "base"
    )

    return tokenizer, sentiment_model, whisper_model


tokenizer, sentiment_model, whisper_model = load_models()

# ==================================
# DOWNLOAD AUDIO
# ==================================

def download_audio(url):

    temp_dir = tempfile.mkdtemp()

    output_template = os.path.join(
        temp_dir,
        "audio.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "extractaudio": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192"
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return os.path.join(
        temp_dir,
        "audio.mp3"
    )

# ==================================
# SENTIMENT
# ==================================

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = sentiment_model(
            **inputs
        )

    probs = F.softmax(
        outputs.logits,
        dim=1
    )

    label = torch.argmax(
        probs
    ).item()

    mapping = {
        0: "positive",
        1: "neutral",
        2: "negative"
    }

    return mapping[label]

# ==================================
# TRANSCRIBE
# ==================================

def transcribe_audio(audio_path):

    result = whisper_model.transcribe(
        audio_path
    )

    return result["text"]

# ==================================
# UI
# ==================================

st.title(
    "🎥 YouTube Sentiment Analyzer"
)

st.write(
    "Paste URL YouTube untuk menganalisis sentimen video."
)

youtube_url = st.text_input(
    "YouTube URL"
)

if st.button("Analisis"):

    if not youtube_url:

        st.warning(
            "Masukkan URL terlebih dahulu."
        )

    else:

        try:

            with st.spinner(
                "Mengunduh audio..."
            ):

                audio_file = download_audio(
                    youtube_url
                )

            with st.spinner(
                "Melakukan transkripsi..."
            ):

                transcript = transcribe_audio(
                    audio_file
                )

            with st.spinner(
                "Menganalisis sentimen..."
            ):

                sentiment = predict_sentiment(
                    transcript
                )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Transcript"
                )

                st.write(
                    transcript[:5000]
                )

            with col2:

                st.subheader(
                    "Sentiment"
                )

                st.metric(
                    "Result",
                    sentiment.upper()
                )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )