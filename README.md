# 🚀 Business Analytics System

> AI-Powered Customer Interaction Analytics Platform using Speech Recognition, NLP, Machine Learning, and Business Intelligence Dashboards.

## 📖 Overview

Business Analytics System is an end-to-end AI-powered platform that analyzes customer call recordings and CRM data to generate actionable business insights.

The system converts audio calls into text using OpenAI Whisper, preprocesses transcripts using spaCy NLP, performs sentiment analysis using a custom TF-IDF + Logistic Regression model or DistilBERT transformer model, stores data in MySQL, and visualizes insights through an interactive Streamlit dashboard.

---

## ✨ Features

### 🎙️ Audio Transcription
- OpenAI Whisper ASR integration
- Supports MP3, WAV, M4A, AAC formats
- GPU acceleration using CUDA
- Automatic transcript caching
- Duplicate transcription prevention

### 🧠 Sentiment Analysis
- TF-IDF + Logistic Regression model
- DistilBERT fallback model
- Sentiment confidence scores
- Cross-validation evaluation
- Real-time predictions

### 🔍 NLP Processing
- Tokenization using spaCy
- Stop-word removal
- Negation preservation
- Custom lemmatization
- Text normalization

### 📊 Analytics Dashboard
- Overall sentiment distribution
- Location-wise sentiment analysis
- Tech stack-wise sentiment analysis
- Interactive charts using Plotly
- Negative keyword extraction

### 💡 Recommendation Engine
Automatically generates recommendations based on:
- Pricing complaints
- Timing issues
- Location concerns
- Placement-related queries
- Learning support concerns

### 🗄️ Database Integration
- MySQL storage
- Transcript management
- Sentiment logs
- Live prediction history

### 📁 Data Export
- Processed dataset download
- Live prediction export
- CSV report generation

---

## 🏗️ System Architecture

```text
Audio Files + CRM CSV
          │
          ▼
   OpenAI Whisper ASR
          │
          ▼
      Transcription
          │
          ▼
      spaCy NLP
(Text Cleaning & Preprocessing)
          │
          ▼
 Sentiment Classification
 ┌───────────────────────┐
 │ TF-IDF + Logistic Reg │
 │      OR               │
 │ DistilBERT Model      │
 └───────────────────────┘
          │
          ▼
      MySQL Database
          │
          ▼
 Analytics Dashboard
          │
          ▼
 Business Recommendations
```

---

## 🛠️ Technology Stack

### Frontend
- Streamlit
- Plotly

### Backend
- Python 3.x

### Database
- MySQL

### Machine Learning
- Scikit-Learn
- Logistic Regression
- TF-IDF Vectorizer

### Deep Learning
- Hugging Face Transformers
- DistilBERT
- PyTorch

### NLP
- spaCy

### Speech Recognition
- OpenAI Whisper

### Data Processing
- Pandas
- NumPy

---

## 📂 Project Structure

```bash
Business-Analytics-System/
│
├── app.py
├── sentiment_model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
│
├── database/
│   └── mysql_schema.sql
│
├── audio_files/
├── processed_data/
└── screenshots/
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/business-analytics-system.git
cd business-analytics-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Install FFmpeg

Whisper requires FFmpeg.

Windows:
Download and add FFmpeg to PATH.

Linux:

```bash
sudo
