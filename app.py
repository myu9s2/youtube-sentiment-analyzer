# =====================================================
# IMPORTS
# =====================================================

import streamlit as st
import yt_dlp
import whisper
import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np

import tempfile
import os
import re

from dotenv import load_dotenv

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sentence_transformers import (
    SentenceTransformer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from googleapiclient.discovery import (
    build
)

from keybert import KeyBERT

from wordcloud import (
    WordCloud,
    STOPWORDS
)

import matplotlib.pyplot as plt

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="🎥",
    layout="wide"
)

# =====================================================
# ENVIRONMENT
# =====================================================

load_dotenv()

YOUTUBE_API_KEY = os.getenv(
    "YOUTUBE_API_KEY"
)

if not YOUTUBE_API_KEY:
    st.error(
        "YOUTUBE_API_KEY tidak ditemukan pada file .env"
    )
    st.stop()

# =====================================================
# MODEL CONFIG
# =====================================================

SENTIMENT_MODEL_NAME = (
    "w11wo/indonesian-roberta-base-sentiment-classifier"
)

SIMILARITY_MODEL_NAME = (
    "paraphrase-multilingual-MiniLM-L12-v2"
)

WHISPER_MODEL_SIZE = (
    "base"
)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    # -----------------------------
    # Sentiment Model
    # -----------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_MODEL_NAME
    )

    sentiment_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            SENTIMENT_MODEL_NAME
        )
    )

    # -----------------------------
    # Whisper
    # -----------------------------

    whisper_model = whisper.load_model(
        WHISPER_MODEL_SIZE
    )

    # -----------------------------
    # Similarity
    # -----------------------------

    similarity_model = (
        SentenceTransformer(
            SIMILARITY_MODEL_NAME
        )
    )

    # -----------------------------
    # Topic Extraction
    # -----------------------------

    keyword_model = KeyBERT()

    return (
        tokenizer,
        sentiment_model,
        whisper_model,
        similarity_model,
        keyword_model
    )

# =====================================================
# INITIALIZE MODELS
# =====================================================

(
    tokenizer,
    sentiment_model,
    whisper_model,
    similarity_model,
    keyword_model
) = load_models()

# =====================================================
# INITIALIZE YOUTUBE API
# =====================================================

youtube = build(
    "youtube",
    "v3",
    developerKey=YOUTUBE_API_KEY
)

# =====================================================
# LABEL MAPPING
# =====================================================

LABELS = {
    0: "positive",
    1: "neutral",
    2: "negative"
}

# =====================================================
# APP TITLE
# =====================================================

st.title(
    "🎥 YouTube Sentiment Analyzer"
)

st.caption(
    "Analisis Sentimen Narasi Video dan Respon Audiens Menggunakan AI"
)

# =====================================================
# YOUTUBE FUNCTIONS
# =====================================================

def get_video_id(url):
    """
    Mendapatkan video id dari berbagai format URL YouTube
    """

    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be\/([a-zA-Z0-9_-]{11})",
        r"shorts\/([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


# =====================================================
# VIDEO METADATA
# =====================================================

def get_video_metadata(video_id):

    request = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=video_id
    )

    response = request.execute()

    if not response["items"]:
        return None

    item = response["items"][0]

    return {
        "title":
            item["snippet"]["title"],

        "channel":
            item["snippet"]["channelTitle"],

        "published":
            item["snippet"]["publishedAt"],

        "views":
            int(
                item["statistics"].get(
                    "viewCount", 0
                )
            ),

        "likes":
            int(
                item["statistics"].get(
                    "likeCount", 0
                )
            ),

        "comments":
            int(
                item["statistics"].get(
                    "commentCount", 0
                )
            )
    }


# =====================================================
# GET COMMENTS
# =====================================================

def get_comments(
    video_id,
    max_comments=500
):

    comments = []

    next_page_token = None

    while len(comments) < max_comments:

        request = (
            youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance"
            )
        )

        response = request.execute()

        for item in response["items"]:

            comment = (
                item["snippet"]
                ["topLevelComment"]
                ["snippet"]
                ["textDisplay"]
            )

            comments.append(comment)

            if len(comments) >= max_comments:
                break

        next_page_token = response.get(
            "nextPageToken"
        )

        if not next_page_token:
            break

    return comments


# =====================================================
# DOWNLOAD AUDIO
# =====================================================

def download_audio(url):

    output_file = "temp_audio.wav"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "temp_audio.%(ext)s",
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key":
                "FFmpegExtractAudio",
                "preferredcodec":
                "wav",
                "preferredquality":
                "192"
            }
        ]
    }

    with yt_dlp.YoutubeDL(
        ydl_opts
    ) as ydl:

        info = ydl.extract_info(
            url,
            download=True
        )

        title = info.get(
            "title",
            "Unknown"
        )

    return output_file, title


# =====================================================
# WHISPER TRANSCRIPTION
# =====================================================

def transcribe_audio(audio_path):

    result = whisper_model.transcribe(
        audio_path,
        language="id"
    )

    return result["text"]


# =====================================================
# SIMPLE TEXT CLEANING
# =====================================================

def clean_text(text):

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# =====================================================
# SENTIMENT PREDICTION
# =====================================================

def predict_sentiment(text):

    text = clean_text(text)

    if not text.strip():
        return "neutral", 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
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

    pred_idx = torch.argmax(
        probs
    ).item()

    confidence = (
        probs[0][pred_idx]
        .item()
    )

    sentiment = LABELS[
        pred_idx
    ]

    return sentiment, confidence


# =====================================================
# SPLIT TRANSCRIPT
# =====================================================

def split_sentences(text):

    sentences = re.split(
        r'[.!?]\s+',
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) > 20:
            cleaned.append(
                sentence
            )

    return cleaned


# =====================================================
# VIDEO SENTIMENT
# =====================================================

def analyze_video_sentiment(
    transcript
):

    sentences = split_sentences(
        transcript
    )

    positive = 0
    neutral = 0
    negative = 0

    for sentence in sentences:

        sentiment, _ = (
            predict_sentiment(
                sentence
            )
        )

        if sentiment == "positive":
            positive += 1

        elif sentiment == "neutral":
            neutral += 1

        else:
            negative += 1

    total = max(
        positive +
        neutral +
        negative,
        1
    )

    return {
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "positive_pct":
            positive / total * 100,
        "neutral_pct":
            neutral / total * 100,
        "negative_pct":
            negative / total * 100
    }


# =====================================================
# COMMENT SENTIMENT
# =====================================================

def analyze_comment_sentiment(
    comments
):

    positive = 0
    neutral = 0
    negative = 0

    positive_comments = []
    negative_comments = []

    for comment in comments:

        sentiment, confidence = (
            predict_sentiment(
                comment
            )
        )

        if sentiment == "positive":

            positive += 1

            positive_comments.append(
                (
                    comment,
                    confidence
                )
            )

        elif sentiment == "neutral":

            neutral += 1

        else:

            negative += 1

            negative_comments.append(
                (
                    comment,
                    confidence
                )
            )

    positive_comments = sorted(
        positive_comments,
        key=lambda x: x[1],
        reverse=True
    )

    negative_comments = sorted(
        negative_comments,
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "top_positive":
            positive_comments[:5],
        "top_negative":
            negative_comments[:5]
    }


# =====================================================
# SIMILARITY
# =====================================================

def calculate_similarity(
    transcript,
    comments
):

    if len(comments) == 0:
        return 0

    comment_text = " ".join(
        comments
    )

    emb_video = (
        similarity_model.encode(
            transcript
        )
    )

    emb_comment = (
        similarity_model.encode(
            comment_text
        )
    )

    similarity = (
        cosine_similarity(
            [emb_video],
            [emb_comment]
        )[0][0]
    )

    return float(
        similarity
    )


# =====================================================
# SIMPLE SUMMARY
# =====================================================

def summarize_text(
    transcript,
    max_sentences=3
):

    sentences = split_sentences(
        transcript
    )

    if len(sentences) <= max_sentences:
        return transcript

    summary = (
        sentences[:max_sentences]
    )

    return ". ".join(
        summary
    ) + "."


# =====================================================
# TOPIC EXTRACTION
# =====================================================

def extract_topics(transcript):

    transcript = transcript[:5000]

    keywords = keyword_model.extract_keywords(
        transcript,
        keyphrase_ngram_range=(2, 4),
        stop_words=None,
        use_maxsum=True,
        nr_candidates=30,
        top_n=8
    )

    topics = []

    blacklist = {
        "yang",
        "dan",
        "untuk",
        "dengan",
        "adalah",
        "karena",
        "dalam",
        "pada",
        "akan",
        "sudah",
        "juga"
    }

    for keyword, score in keywords:

        keyword = keyword.strip()

        if keyword.lower() in blacklist:
            continue

        topics.append(keyword)

    return topics


# =====================================================
# INTERPRETATION
# =====================================================

def generate_insight(
    video_sentiment,
    comment_sentiment,
    similarity
):

    dominant_video = max(
        [
            (
                "positif",
                video_sentiment["positive"]
            ),
            (
                "netral",
                video_sentiment["neutral"]
            ),
            (
                "negatif",
                video_sentiment["negative"]
            )
        ],
        key=lambda x: x[1]
    )[0]

    dominant_comment = max(
        [
            (
                "positif",
                comment_sentiment["positive"]
            ),
            (
                "netral",
                comment_sentiment["neutral"]
            ),
            (
                "negatif",
                comment_sentiment["negative"]
            )
        ],
        key=lambda x: x[1]
    )[0]

    if similarity > 0.80:
        relation = (
            "Audiens membahas topik "
            "yang sangat relevan "
            "dengan isi video."
        )

    elif similarity > 0.60:
        relation = (
            "Audiens membahas topik "
            "yang cukup relevan "
            "dengan isi video."
        )

    else:
        relation = (
            "Komentar audiens "
            "cenderung keluar dari "
            "topik utama video."
        )

    insight = f"""
    Sentimen dominan video adalah {dominant_video}.

    Sentimen dominan komentar adalah {dominant_comment}.

    Nilai similarity sebesar {similarity:.2f}.

    {relation}
    """

    return insight


# =====================================================
# PIE CHART
# =====================================================

def plot_sentiment_pie(
    positive,
    neutral,
    negative,
    title
):

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    labels = [
        "Positive",
        "Neutral",
        "Negative"
    ]

    values = [
        positive,
        neutral,
        negative
    ]

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%"
    )

    ax.set_title(title)

    return fig


# =====================================================
# BAR CHART
# =====================================================

def plot_comparison_bar(
    video_sentiment,
    comment_sentiment
):

    labels = [
        "Positive",
        "Neutral",
        "Negative"
    ]

    video_values = [
        video_sentiment["positive"],
        video_sentiment["neutral"],
        video_sentiment["negative"]
    ]

    comment_values = [
        comment_sentiment["positive"],
        comment_sentiment["neutral"],
        comment_sentiment["negative"]
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x - width / 2,
        video_values,
        width,
        label="Video"
    )

    ax.bar(
        x + width / 2,
        comment_values,
        width,
        label="Komentar"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels
    )

    ax.set_title(
        "Perbandingan Sentimen"
    )

    ax.legend()

    return fig


# =====================================================
# WORD CLOUD
# =====================================================

def generate_wordcloud(comments):

    text = " ".join(comments)

    if not text.strip():
        return None

    indonesia_stopwords = {
        "yang",
        "dan",
        "untuk",
        "ini",
        "itu",
        "dengan",
        "dari",
        "pada",
        "karena",
        "juga",
        "ada",
        "akan",
        "sudah",
        "saja",
        "nya",
        "jadi",
        "atau",
        "agar",
        "dalam",
        "lebih",
        "masih",
        "bisa",
        "harus",
        "semua",
        "saya",
        "kami",
        "kita",
        "mereka"
    }

    stopwords = STOPWORDS.union(
        indonesia_stopwords
    )

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        stopwords=stopwords,
        collocations=False
    ).generate(text)

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.imshow(
        wc,
        interpolation="bilinear"
    )

    ax.axis("off")

    return fig


# =====================================================
# SIMILARITY INTERPRETATION
# =====================================================

def similarity_label(
    similarity
):

    if similarity >= 0.80:
        return "Tinggi"

    elif similarity >= 0.60:
        return "Sedang"

    else:
        return "Rendah"
    

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Pengaturan")

youtube_url = st.sidebar.text_input(
    "URL YouTube"
)

comment_limit = st.sidebar.selectbox(
    "Jumlah Komentar",
    [100, 250, 500, 1000],
    index=2
)

analyze_btn = st.sidebar.button(
    "🚀 Analisis"
)

# =====================================================
# MAIN PROCESS
# =====================================================

if analyze_btn:

    if not youtube_url:

        st.warning(
            "Masukkan URL YouTube terlebih dahulu."
        )

        st.stop()

    try:

        # =====================================
        # VIDEO ID
        # =====================================

        with st.spinner(
            "Mengambil metadata video..."
        ):

            video_id = get_video_id(
                youtube_url
            )

            if not video_id:

                st.error(
                    "URL YouTube tidak valid."
                )

                st.stop()

            metadata = get_video_metadata(
                video_id
            )

        # =====================================
        # METADATA
        # =====================================

        st.header("📺 Informasi Video")

        st.subheader(metadata["title"])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "👤 Channel",
                metadata["channel"]
            )

        with col2:
            st.metric(
                "👁️ Views",
                f"{metadata['views']:,}"
            )

        with col3:
            st.metric(
                "👍 Likes",
                f"{metadata['likes']:,}"
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "💬 Komentar",
                f"{metadata['comments']:,}"
            )

        with col2:
            st.metric(
                "📅 Upload",
                metadata["published"][:10]
            )

        # =====================================
        # COMMENTS
        # =====================================

        with st.spinner(
            "Mengambil komentar..."
        ):

            comments = get_comments(
                video_id,
                comment_limit
            )

        st.success(
            f"{len(comments)} komentar berhasil diambil."
        )

        # =====================================
        # AUDIO
        # =====================================

        with st.spinner(
            "Mengunduh audio..."
        ):

            audio_file, title = (
                download_audio(
                    youtube_url
                )
            )

        # =====================================
        # TRANSCRIPT
        # =====================================

        with st.spinner(
            "Melakukan transkripsi..."
        ):

            transcript = (
                transcribe_audio(
                    audio_file
                )
            )

        # =====================================
        # SUMMARY
        # =====================================

        summary = summarize_text(
            transcript
        )

        # =====================================
        # TOPICS
        # =====================================

        topics = extract_topics(
            transcript
        )

        # =====================================
        # VIDEO SENTIMENT
        # =====================================

        with st.spinner(
            "Analisis sentimen video..."
        ):

            video_sentiment = (
                analyze_video_sentiment(
                    transcript
                )
            )

        # =====================================
        # COMMENT SENTIMENT
        # =====================================

        with st.spinner(
            "Analisis sentimen komentar..."
        ):

            comment_sentiment = (
                analyze_comment_sentiment(
                    comments
                )
            )

        # =====================================
        # SIMILARITY
        # =====================================

        similarity = (
            calculate_similarity(
                transcript,
                comments
            )
        )

        # =====================================
        # INSIGHT
        # =====================================

        insight = generate_insight(
            video_sentiment,
            comment_sentiment,
            similarity
        )

        # =====================================
        # SUMMARY
        # =====================================

        st.header(
            "📝 Ringkasan Video"
        )

        st.write(summary)

        # =====================================
        # TOPICS
        # =====================================

        st.header("🏷️ Topik Utama")

        cols = st.columns(4)

        for i, topic in enumerate(topics):

            cols[i % 4].info(topic)

        # =====================================
        # TRANSCRIPT
        # =====================================

        with st.expander(
            "Lihat Transkrip Lengkap"
        ):

            st.write(
                transcript
            )

        # =====================================
        # VIDEO SENTIMENT
        # =====================================

        st.header(
            "🎥 Sentimen Video"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Positif",
            video_sentiment["positive"]
        )

        col2.metric(
            "Netral",
            video_sentiment["neutral"]
        )

        col3.metric(
            "Negatif",
            video_sentiment["negative"]
        )

        st.pyplot(
            plot_sentiment_pie(
                video_sentiment["positive"],
                video_sentiment["neutral"],
                video_sentiment["negative"],
                "Sentimen Video"
            )
        )

        # =====================================
        # COMMENT SENTIMENT
        # =====================================

        st.header(
            "💬 Sentimen Komentar"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Positif",
            comment_sentiment["positive"]
        )

        col2.metric(
            "Netral",
            comment_sentiment["neutral"]
        )

        col3.metric(
            "Negatif",
            comment_sentiment["negative"]
        )

        st.pyplot(
            plot_sentiment_pie(
                comment_sentiment["positive"],
                comment_sentiment["neutral"],
                comment_sentiment["negative"],
                "Sentimen Komentar"
            )
        )

        # =====================================
        # COMPARISON
        # =====================================

        st.header(
            "📊 Perbandingan Sentimen"
        )

        st.pyplot(
            plot_comparison_bar(
                video_sentiment,
                comment_sentiment
            )
        )

        # =====================================
        # SIMILARITY
        # =====================================

        st.header(
            "🔗 Similarity"
        )

        st.metric(
            "Similarity Score",
            f"{similarity:.2f}"
        )

        st.info(
            f"Tingkat Similarity: "
            f"{similarity_label(similarity)}"
        )

        # =====================================
        # WORD CLOUD
        # =====================================

        st.header(
            "☁️ Word Cloud Komentar"
        )

        wc = generate_wordcloud(
            comments
        )

        if wc:
            st.pyplot(wc)

        # =====================================
        # INSIGHT
        # =====================================

        st.header(
            "🧠 Insight"
        )

        st.info(insight)

        # =====================================
        # TOP POSITIVE
        # =====================================

        st.header(
            "😊 Top Positive Comments"
        )

        for comment, score in (
            comment_sentiment[
                "top_positive"
            ]
        ):

            st.success(
                comment
            )

        # =====================================
        # TOP NEGATIVE
        # =====================================

        st.header(
            "😡 Top Negative Comments"
        )

        for comment, score in (
            comment_sentiment[
                "top_negative"
            ]
        ):

            st.error(
                comment
            )

        # =====================================
        # CLEANUP
        # =====================================

        if os.path.exists(
            audio_file
        ):

            os.remove(
                audio_file
            )

    except Exception as e:

        st.exception(e)