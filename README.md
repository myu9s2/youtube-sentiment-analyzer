# 🎥 YouTube Sentiment Analyzer

Aplikasi berbasis Streamlit untuk menganalisis sentimen narasi video YouTube dan membandingkannya dengan sentimen audiens melalui komentar menggunakan teknologi Natural Language Processing (NLP) dan Artificial Intelligence (AI).

## Fitur Utama

### Analisis Video

* Mengambil audio dari video YouTube menggunakan yt-dlp
* Transkripsi otomatis menggunakan OpenAI Whisper
* Ringkasan isi video secara otomatis
* Ekstraksi topik utama menggunakan KeyBERT
* Analisis sentimen narasi video menggunakan RoBERTa Bahasa Indonesia

### Analisis Komentar

* Mengambil komentar langsung dari YouTube Data API v3
* Mendukung pengambilan hingga 1000 komentar
* Analisis sentimen komentar (Positive, Neutral, Negative)
* Menampilkan komentar positif dan negatif dominan

### Perbandingan Sentimen

* Membandingkan sentimen narasi video dengan sentimen audiens
* Menghitung tingkat kemiripan (Cosine Similarity) antara isi video dan komentar
* Insight otomatis berdasarkan hasil analisis

### Visualisasi

* Pie Chart distribusi sentimen video
* Pie Chart distribusi sentimen komentar
* Bar Chart perbandingan sentimen
* Word Cloud komentar

---

## Teknologi yang Digunakan

### Natural Language Processing

* OpenAI Whisper
* RoBERTa Indonesian Sentiment Classifier
* Sentence Transformers
* KeyBERT

### Data Processing

* Pandas
* NumPy
* Scikit-Learn

### Visualization

* Matplotlib
* WordCloud

### Deployment & Interface

* Streamlit
* YouTube Data API v3
* yt-dlp

---

## Arsitektur Sistem

```text
YouTube URL
        │
        ▼
 YouTube API
 ├── Metadata
 └── Comments
        │
        ▼
     yt-dlp
        │
        ▼
 Audio Extraction
        │
        ▼
 OpenAI Whisper
        │
        ▼
   Transcript
        │
        ├─────────────► Summary
        │
        ├─────────────► Topic Extraction
        │
        ├─────────────► Video Sentiment
        │
        ▼
 Comment Sentiment
        │
        ▼
 Similarity Analysis
        │
        ▼
 Dashboard & Visualization
```

---

## Struktur Proyek

```text
youtube-sentiment-analyzer/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
│
└── assets/
```

---

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/myu9s2/youtube-sentiment-analyzer
cd youtube-sentiment-analyzer
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Konfigurasi API

Buat file `.env` pada root project:

```env
YOUTUBE_API_KEY=YOUR_YOUTUBE_API_KEY
```

### Mendapatkan API Key

1. Buka Google Cloud Console
2. Buat project baru
3. Aktifkan YouTube Data API v3
4. Buat API Key
5. Simpan API Key ke file `.env`

---

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan tersedia pada:

```text
http://localhost:8501
```

---

## Output Analisis

Aplikasi menghasilkan:

* Informasi video
* Ringkasan video
* Topik utama
* Transkrip video
* Sentimen video
* Sentimen komentar
* Similarity score
* Insight otomatis
* Word Cloud
* Pie Chart
* Bar Chart
* Top Positive Comments
* Top Negative Comments

---

## Model yang Digunakan

### Sentiment Analysis

Model:

```text
w11wo/indonesian-roberta-base-sentiment-classifier
```

### Similarity Analysis

Model:

```text
paraphrase-multilingual-MiniLM-L12-v2
```

### Speech-to-Text

Model:

```text
OpenAI Whisper (base)
```

---

## Catatan

* Pengunduhan model pertama kali membutuhkan koneksi internet.
* Startup pertama mungkin memerlukan waktu lebih lama karena model AI akan diunduh dan disimpan ke cache lokal.
* Jumlah komentar yang dianalisis memengaruhi waktu proses.
* Video dengan durasi panjang memerlukan waktu transkripsi lebih lama.

---

## Pengembang

Yuga
Informatics Student at Siliwangi University | Junior Data Science Intern at Vinix7

Project ini dikembangkan sebagai implementasi Natural Language Processing (NLP), Sentiment Analysis, Topic Extraction, dan Similarity Analysis pada platform YouTube.
