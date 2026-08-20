 #Imports & Config
import os
import io
import tempfile
from datetime import datetime
import pickle
import joblib
import re
import torch
import spacy
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
from transformers import pipeline

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import mysql.connector
from mysql.connector import errorcode

# Optional whisper import (may be None)
try:
    import whisper
except Exception:
    whisper = None

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "56ya6ay8@",
    "database": "banalytics"
}

# File names for model persistence
VECTORIZER_PATH = "vectorizer.pkl"
MODEL_PATH = "sentiment_model.pkl"

# Streamlit basic UI config
st.set_page_config(page_title="Business Analytics System ", layout="wide")
st.title("Business Analytics System ")
st.caption("Audio → Transcripts → Sentiment → DB storage → Export")

# MySQL helpers

def connect_db():
    """Return a mysql.connector connection using DB_CONFIG."""
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG.get("port", 3306),
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            autocommit=True
        )
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            # Database doesn't exist
            raise RuntimeError("Database not found. Create database 'banalytics' first or run the CREATE DB SQL.")
        else:
            raise

def ensure_tables_exist():
    """Create tables if they do not exist."""
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG.get("port", 3306),
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        autocommit=True
    )
    cur = conn.cursor()
    # Create database if not exists
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}`;")
    cur.execute(f"USE `{DB_CONFIG['database']}`;")
    # transcripts
    cur.execute("""
    CREATE TABLE IF NOT EXISTS transcripts (
        id INT AUTO_INCREMENT PRIMARY KEY,
        call_id VARCHAR(255),
        transcript LONGTEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY(call_id)
    );
    """)
    # live_predictions
    cur.execute("""
    CREATE TABLE IF NOT EXISTS live_predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        call_id VARCHAR(255),
        transcript LONGTEXT,
        prediction VARCHAR(50),
        score FLOAT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    # call_logs (merged)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS call_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        call_id VARCHAR(255),
        student_name VARCHAR(255),
        year VARCHAR(50),
        tech_stack VARCHAR(255),
        location VARCHAR(255),
        remarks LONGTEXT,
        transcript_text LONGTEXT,
        combined_text LONGTEXT,
        cleaned_text LONGTEXT,
        label VARCHAR(50),
        sentiment VARCHAR(50),
        sentiment_score FLOAT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY(call_id)
    );
    """)
    cur.close()
    conn.close()

# Run ensure at app start (safe)
try:
    ensure_tables_exist()
except Exception as e:
    st.error(f"Database setup error: {e}")
    st.stop()

# Model persistence & training helpers
from sklearn.exceptions import NotFittedError

@st.cache_resource(show_spinner=False)
def load_saved_model():
    """Try to load saved vectorizer and classifier from disk."""
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH):
        try:
            vectorizer = joblib.load(VECTORIZER_PATH)
            clf = joblib.load(MODEL_PATH)
            st.sidebar.success("Sentiment model & vectorizer loaded from disk.")
            return vectorizer, clf
        except Exception as e:
            st.sidebar.warning(f"Failed to load saved model: {e}")
            return None
    return None

def save_model_to_disk(vectorizer, clf):
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(clf, MODEL_PATH)
    st.success("Saved vectorizer and model to disk.")


@st.cache_resource(show_spinner="Loading spaCy model...")
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
    except OSError:
        st.error(f"spaCy model '{model_name}' not found. Please run: python -m spacy download {model_name}")
        nlp = spacy.blank("en")
    return nlp

#remove stop words and skip these negations
nlp = load_spacy_model()
NEGATIONS = {"not", "no", "never", "none", "nor", "cannot", "can't", "won't", "don't", "dont"}
base_stopwords = nlp.Defaults.stop_words
custom_stop_words = set(base_stopwords) - NEGATIONS
LEMMA_CORRECTIONS = {
    "interested": "interest",
    "considering": "consider",
    "considered": "consider",
    "fees": "fee",
    "payments": "payment",
    "courses": "course",
    "students": "student",
    "options": "option",
}


def spacy_lemmatizer_tokenizer(text: str):
    if not isinstance(text, str) or text.strip() == "":
        return []
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower().strip()
        if token.is_punct or token.like_num or not token.is_alpha or lemma == "-pron-":
            continue
        if lemma in NEGATIONS:
            tokens.append(lemma)
            continue
        if lemma in custom_stop_words:
            continue
        if lemma in LEMMA_CORRECTIONS:
            lemma = LEMMA_CORRECTIONS[lemma]
        tokens.append(lemma)
    return tokens

#cross validation score and return mean score in report for accuracy
@st.cache_data(show_spinner="Running cross-validation...")
def get_cv_report(texts: pd.Series, labels: pd.Series, cv: int = 5) -> pd.DataFrame:
    """
    Run k-fold cross validation (default 5-fold) and return a summary
    of accuracy / precision / recall / f1 (weighted).
    """
    # 1) Clean labels exactly like before
    y = labels.astype(str).str.lower().replace({
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
        "n": "negative",
        "p": "positive"
    })

    # 2) Pipeline = TF-IDF + Logistic Regression
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2),tokenizer=spacy_lemmatizer_tokenizer),
        LogisticRegression(max_iter=200)
    )

    # 3) Define which metrics to compute on each fold
    scoring = {
    "accuracy": "accuracy",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "f1_weighted": "f1_weighted",
}

    cv_results = cross_validate(
        model,
        texts,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    metrics_dict = {
        "accuracy": cv_results["test_accuracy"],
        "precision_weighted": cv_results["test_precision_weighted"],
        "recall_weighted": cv_results["test_recall_weighted"],
        "f1_weighted": cv_results["test_f1_weighted"],
    }

    rows = []
    for metric_name, values in metrics_dict.items():
        values = np.array(values, dtype=float)
        rows.append({
            "metric": metric_name,
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
        })

    summary_df = pd.DataFrame(rows).set_index("metric")
    return summary_df

# final model for saving and predictions when rerun the project it will be loaded from disk
@st.cache_resource(show_spinner="Training final sentiment model...")
def train_final_model(texts: pd.Series, labels: pd.Series):
    y = labels.astype(str).str.lower().replace({"pos":"positive","neg":"negative","neu":"neutral","n":"negative","p":"positive"})
    vectorizer = TfidfVectorizer(ngram_range=(1, 2),tokenizer=spacy_lemmatizer_tokenizer)
    Xtr = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, y)
    return vectorizer, clf

def predict_sentiment(vectorizer, clf, raw_text: str):
    if not raw_text or not isinstance(raw_text, str) or raw_text.strip() == "":
        return "neutral", 0.5
    try:
        text_vector = vectorizer.transform([raw_text])
        prediction = clf.predict(text_vector)[0]
        proba = float(np.max(clf.predict_proba(text_vector)))
        return prediction, proba
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "neutral", 0.0

# Load saved model at app start if present
saved = load_saved_model()
if saved:
    st.session_state["sentiment_model"] = saved
    st.session_state["model_trained"] = True

# Whisper load & transcription
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str):
    if whisper is None:
        raise RuntimeError("Whisper not installed. pip install openai-whisper and ensure ffmpeg is present.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    st.sidebar.success(f"Whisper will run on: {device.upper()}")
    return whisper.load_model(model_size, device=device)

def transcribe_with_whisper(audio_bytes: bytes, model, filename: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name
    try:
        result = model.transcribe(path,language="en",
    task="transcribe")
        return result.get("text", "").strip()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

#pretrained model of distilbert trained on millions of data
@st.cache_resource(show_spinner=False)
def load_hf_pipeline():
    # Fast, widely used binary sentiment model
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


# DB CRUD for transcripts & logs
def get_transcript_from_db(call_id: str):
    """Return transcript text or None."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT transcript FROM transcripts WHERE call_id=%s LIMIT 1", (call_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None

def save_transcript_to_db(call_id: str, transcript_text: str):
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO transcripts (call_id, transcript) VALUES (%s,%s) ON DUPLICATE KEY UPDATE transcript=%s, created_at=CURRENT_TIMESTAMP", (call_id, transcript_text, transcript_text))
        conn.commit()
    finally:
        cur.close()
        conn.close()

def save_live_prediction_db(call_id: str, transcript_text: str, prediction: str, score: float):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO live_predictions (call_id, transcript, prediction, score) VALUES (%s,%s,%s,%s)", (call_id, transcript_text, prediction, score))
    conn.commit()
    cur.close()
    conn.close()

def upsert_call_log(row: dict):
  
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO call_logs
        (call_id, student_name, year, tech_stack, location, remarks, transcript_text, combined_text, cleaned_text, label, sentiment, sentiment_score)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            student_name=VALUES(student_name),
            year=VALUES(year),
            tech_stack=VALUES(tech_stack),
            location=VALUES(location),
            remarks=VALUES(remarks),
            transcript_text=VALUES(transcript_text),
            combined_text=VALUES(combined_text),
            cleaned_text=VALUES(cleaned_text),
            label=VALUES(label),
            sentiment=VALUES(sentiment),
            sentiment_score=VALUES(sentiment_score),
            created_at=CURRENT_TIMESTAMP
    """, (
        row.get("call_id"),
        row.get("student_name"),
        row.get("year"),
        row.get("tech_stack"),
        row.get("location"),
        row.get("remarks"),
        row.get("transcript_text"),
        row.get("combined_text"),
        row.get("cleaned_text"),
        row.get("label"),
        row.get("sentiment"),
        row.get("sentiment_score")
    ))
    conn.commit()
    cur.close()
    conn.close()

def save_merged_sentiment_row(row):
    conn = connect_db()
    cur = conn.cursor()
    sql = """INSERT INTO merged_sentiment_logs
        (call_id, student_name, year, tech_stack, location, remarks, label, transcript_text, cleaned_text, sentiment, sentiment_score)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    cur.execute(sql, (
        row.get("call_id"),
        row.get("student_name"),
        row.get("year"),
        row.get("tech_stack"),
        row.get("location"),
        row.get("remarks"),
        row.get("label"),
        row.get("transcript_text"),
        row.get("cleaned_text"),
        row.get("sentiment"),
        row.get("sentiment_score"),
    ))
    conn.commit()
    cur.close()
    conn.close()

def get_all_transcripts_from_db():
    conn = connect_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT call_id, transcript AS transcript_text FROM transcripts ORDER BY created_at DESC;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows)

# Streamlit UI - Upload & Transcribe & Merge
st.sidebar.header("Settings")
asr_engine = st.sidebar.selectbox("ASR Engine (Audio → Text)", ["Whisper"])
whisper_size = st.sidebar.selectbox("Whisper model size", ["base", "small", "medium"],index=0)

use_hf = st.sidebar.checkbox(
    "Force HuggingFace model for sentiment",
    value=False,
    help="Use HF DistilBERT sentiment model even if CSV has labels."
)

st.sidebar.markdown("---")
save_intermediate = st.sidebar.checkbox("Save processed CSV", value=True)

st.subheader("1) Upload Data")
col1, col2 = st.columns([1,1])
with col1:
    csv_file = st.file_uploader("Upload CSV logs (remarks, student, year, tech stack, location, label)", type=["csv"])
with col2:
    audio_files = st.file_uploader("Upload call recordings (mp3/wav/m4a/aac)", type=["mp3","wav","m4a","aac"], accept_multiple_files=True)

# Option: transcribe only new audio (skips IDs found in transcripts table)
transcribe_only_new = st.checkbox("Transcribe ONLY new audio files (skip call_ids already in DB)", value=True)

# Prepare whisper model if needed
whisper_model = None
if audio_files and asr_engine == "Whisper":
    try:
        whisper_model = load_whisper(whisper_size)
    except Exception as e:
        st.error(f"Whisper load error: {e}")
        whisper_model = None

# Build transcripts list but check DB first
transcripts = []
if audio_files:
    st.subheader("2) Transcription (DB-aware)")
    prog = st.progress(0)
    for i, f in enumerate(audio_files):
        cid = os.path.splitext(os.path.basename(f.name))[0]
        existing = get_transcript_from_db(cid)
        if existing and transcribe_only_new:
            # use DB copy
            transcripts.append({"call_id": cid, "transcript_text": existing, "from_db": True})
        else:
            # transcribe and save
            audio_bytes = f.read()
            if asr_engine == "Whisper" and whisper_model is not None:
                text = transcribe_with_whisper(audio_bytes, whisper_model, f.name)
            else:
                text = "[ASR not available]"
            transcripts.append({"call_id": cid, "transcript_text": text, "from_db": False})
            save_transcript_to_db(cid, text)
        prog.progress(int(((i+1)/len(audio_files))*100))
    st.success(f"Processed {len(transcripts)} audio file(s)")
else:
    # keep empty DataFrame structure
    transcripts = []

# Convert to DataFrame
if transcripts:
    new_df = pd.DataFrame(transcripts).drop_duplicates(subset=["call_id"]).reset_index(drop=True)
else:
    new_df = pd.DataFrame(columns=["call_id", "transcript_text"])

# Fetch existing transcripts from DB
try:
    db_df = get_all_transcripts_from_db()
except Exception:
    db_df = pd.DataFrame(columns=["call_id", "transcript_text", "cleaned_text"])

# Merge new + old and drop duplicates
df_tr = pd.concat([new_df, db_df], ignore_index=True).drop_duplicates(subset=["call_id"], keep="first")
# keep all live-prediction rows in this browser session
if "live_rows" not in st.session_state:
    st.session_state["live_rows"] = []  # list of dicts, each like row_for_log


# Merge CSV + transcripts (only run when CSV uploaded as you requested)
merged = None
if csv_file is not None:
    try:
        df_raw = pd.read_csv(csv_file)
    except Exception:
        df_raw = pd.read_csv(csv_file, encoding="latin-1")
    st.success(f"CSV loaded with shape {df_raw.shape}")

    # column mapping UI (same as original)
    with st.expander("Map columns (flexible)"):
        cols = ["<none>"] + list(df_raw.columns)
        map_student = st.selectbox("Student Name column", cols, index=cols.index("student_name") if "student_name" in df_raw.columns else 0)
        map_stack = st.selectbox("Tech Stack column", cols, index=cols.index("tech_stack") if "tech_stack" in df_raw.columns else 0)
        map_loc = st.selectbox("Location column", cols, index=cols.index("location") if "location" in df_raw.columns else 0)
        map_remarks = st.selectbox("Remarks/Notes column", cols, index=cols.index("remarks") if "remarks" in df_raw.columns else 0)
        map_callid = st.selectbox("Call ID column ", cols, index=cols.index("call_id") if "call_id" in df_raw.columns else 0)
        map_label = st.selectbox("Sentiment label column (optional: positive/neutral/negative)", cols, index=cols.index("sentiment_label") if "sentiment_label" in df_raw.columns else 0)

    def pick(colname):
        return None if colname == "<none>" else df_raw[colname]

    df = pd.DataFrame({
        "call_id": pick(map_callid) if map_callid != "<none>" else pd.Series([None]*len(df_raw)),
        "student_name": pick(map_student) if map_student != "<none>" else pd.Series([None]*len(df_raw)),
        "tech_stack": pick(map_stack) if map_stack != "<none>" else pd.Series([None]*len(df_raw)),
        "location": pick(map_loc) if map_loc != "<none>" else pd.Series([None]*len(df_raw)),
        "remarks": pick(map_remarks) if map_remarks != "<none>" else pd.Series([""]*len(df_raw)),
        "label": pick(map_label) if map_label != "<none>" else pd.Series([None]*len(df_raw)),
    })

    # normalize call_id to string
    if "call_id" in df.columns:
        df["call_id"] = df["call_id"].astype(str).str.replace(r'\.0$', '', regex=True)
    # Ensure transcript df_tr fetched above (from uploads/DB)
    if not df_tr.empty:
        df_tr["call_id"] = df_tr["call_id"].astype(str)
        merged = pd.merge(df, df_tr, on="call_id", how="left")
    else:
        # If no audio uploaded, still we want to try fetching transcripts from DB for call_ids in CSV
        # fetch transcripts for all CSV call_ids
        csv_call_ids = df["call_id"].dropna().unique().tolist()
        rows = []
        for cid in csv_call_ids:
            t = get_transcript_from_db(cid)
            rows.append({"call_id": cid, "transcript_text": t if t is not None else ""})
        db_tr_df = pd.DataFrame(rows)
        merged = pd.merge(df, db_tr_df, on="call_id", how="left")
    
    if st.session_state["live_rows"]:
        merged = pd.concat(
            [merged, pd.DataFrame(st.session_state["live_rows"])],
            ignore_index=True
        )

    merged["remarks"] = merged.get("remarks", pd.Series([""] * len(merged))).fillna("")
    merged["transcript_text"] = merged.get("transcript_text", "").fillna("")
    merged["combined_text"] = (merged["remarks"].astype(str) + " " + merged["transcript_text"].astype(str)).str.strip()
    merged["cleaned_text"] = merged["combined_text"].apply(lambda x: " ".join(spacy_lemmatizer_tokenizer(str(x))))
    
    # Save merged rows into call_logs table (upsert)
    for _, row in merged.iterrows():
        upsert_call_log({
            "call_id": row.get("call_id"),
            "student_name": row.get("student_name"),
            "tech_stack": row.get("tech_stack"),
            "location": row.get("location"),
            "remarks": row.get("remarks"),
            "transcript_text": row.get("transcript_text"),
            "combined_text": row.get("combined_text"),
            "cleaned_text": row.get("cleaned_text"),
            "label": row.get("label"),
        })

    # Show merged results (only rows with transcript or show all if none)
    df_display = merged[merged['transcript_text'].notna() & (merged['transcript_text'] != '')]
    if df_display.empty:
        st.info("No transcripts found for CSV call_ids (or audio not uploaded). Showing the merged CSV data including cleaned text.")
        st.dataframe(merged.head(50), width='stretch', hide_index=True)
    else:
        st.write(f"Showing {len(df_display)} merged row(s) that have a transcript (including cleaned text):")
        st.dataframe(df_display.head(50), width='stretch', hide_index=True)

can_train = (merged is not None) and ("label" in merged.columns) and merged["label"].notna().any()

# custom train model if target column present and calculate accuracy
if can_train:
    st.info("Labels found in CSV – you can evaluate model with cross-validation.")

    train_data = merged[merged["label"].notna()]

    st.write("### Step 1: Evaluate Model (Cross-Validation)")
    if st.sidebar.button("Run Cross-Validation"):
        cv_summary = get_cv_report(
            train_data["combined_text"],
            train_data["label"],
            cv=6,   
        )

        st.success("Cross-validation complete (5 folds).")
        st.caption("Scores are mean ± std across folds. Higher is better.")
        st.dataframe(
            cv_summary.style.format("{:.3f}"),
            use_container_width=True
        )

    st.write("### Step 2: Train Final Model")
    if st.sidebar.button("Train Final Model for Prediction"):
        vectorizer, clf = train_final_model(
            train_data["combined_text"],
            train_data["label"]
        )
        st.session_state["sentiment_model"] = (vectorizer, clf)
        st.session_state["model_trained"] = True
        save_model_to_disk(vectorizer, clf)
        st.success("✅ Final sentiment model trained & saved.")
else:
    st.warning("No valid label column found. Accuracy cannot be calculated.")


# Train & Apply sentiment - saves predictions into call_logs table
# -----------------------------------------
# SENTIMENT DECISION LOGIC (custom vs HF)
# -----------------------------------------

if merged is not None and len(merged) > 0:

    st.subheader(" Apply Sentiment & Analyze")

    # 1️⃣ HuggingFace toggle ON → always use HF DistilBERT
    if use_hf:
        st.warning("Using HuggingFace DistilBERT (forced via checkbox).")
        with st.spinner("Running HuggingFace sentiment model…"):
            nlp_ob = load_hf_pipeline()
            preds, scores = [], []

            for txt in merged["combined_text"].fillna(""):
                try:
                    r = nlp_ob(txt[:4096])[0]
                    label = r["label"].lower()

                    if label == "positive":
                        preds.append("positive")
                    elif label == "negative":
                        preds.append("negative")
                    else:
                        preds.append("neutral")

                    scores.append(float(r.get("score", 0)))
                except:
                    preds.append("neutral")
                    scores.append(0.0)

        merged["sentiment"] = preds
        merged["sentiment_score"] = scores
        model_used = "huggingface"

    # 2️⃣ Use custom trained model (if labels exist AND HF toggle off)
    elif st.session_state.get("model_trained"):
        st.success("Using custom TF-IDF model (trained from CSV labels).")

        vectorizer, clf = st.session_state["sentiment_model"]
        X_all = vectorizer.transform(merged["combined_text"])
        probas = clf.predict_proba(X_all)

        merged["sentiment"] = clf.predict(X_all)
        merged["sentiment_score"] = np.max(probas, axis=1)
        model_used = "custom_tf_idf"

    # 3️⃣ No labels + HF toggle off → fallback to HF model
    else:
        st.warning("No labels found → Using HuggingFace fallback model.")
        with st.spinner("Running HuggingFace sentiment model…"):
            nlp_ob = load_hf_pipeline()
            preds, scores = [], []

            for txt in merged["combined_text"].fillna(""):
                try:
                    r = nlp_ob(txt[:4096])[0]
                    label = r["label"].lower()

                    if label == "positive":
                        preds.append("positive")
                    elif label == "negative":
                        preds.append("negative")
                    else:
                        preds.append("neutral")

                    scores.append(float(r.get("score", 0)))
                except:
                    preds.append("neutral")
                    scores.append(0.0)

        merged["sentiment"] = preds
        merged["sentiment_score"] = scores
        model_used = "huggingface_fallback"

    st.success(f"Sentiment applied using: {model_used}")

    # -----------------------------
    # Save sentiment back to DB
    # -----------------------------
    for _, r in merged.iterrows():
        upsert_call_log({
            "call_id": r.get("call_id"),
            "student_name": r.get("student_name"),
            "tech_stack": r.get("tech_stack"),
            "location": r.get("location"),
            "remarks": r.get("remarks"),
            "transcript_text": r.get("transcript_text"),
            "combined_text": r.get("combined_text"),
            "cleaned_text": r.get("cleaned_text"),
            "label": r.get("label"),
            "sentiment": r.get("sentiment"),
            "sentiment_score": float(r.get("sentiment_score"))
                    if r.get("sentiment_score") is not None else None
        })
    for _, r in merged.iterrows():
        try:
            save_merged_sentiment_row(r)
        except Exception as e:
            st.warning(f"Could not save row for call_id {r.get('call_id')}: {e}")

    # (The rest of your analytics UI kept - charts, keywords)
    st.markdown("### Analytics")
    colA, colB, colC = st.columns(3)
    with colA:
        fig = px.pie(merged, names="sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
    # -----------------------------
    # Sentiment Pie Charts by Location (Clean Layout)
    # -----------------------------
        st.subheader("Sentiment Distribution by Location (Pie Charts)")

        if "location" in merged.columns and "sentiment" in merged.columns:
            merged["location"] = merged["location"].astype(str).str.strip().str.title()
            locations = merged["location"].dropna().unique()

            for loc in locations:
                df_loc = merged[merged["location"] == loc]

                pos = (df_loc["sentiment"] == "positive").sum()
                neg = (df_loc["sentiment"] == "negative").sum()
                total = pos + neg

                if total == 0:
                    continue

                pos_pct = (pos / total) * 100
                neg_pct = (neg / total) * 100

                st.write(f"### {loc} — Positive: {pos_pct:.1f}% | Negative: {neg_pct:.1f}%")
                # Make the pie chart
                fig = px.pie(
                    names=["Positive", "Negative"],
                    values=[pos, neg],
                    hole=0.35,
                )
                fig.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    margin=dict(t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No location or sentiment data available.")


    with colC:
        if "tech_stack" in merged.columns:
            fig3 = px.bar(merged.fillna({"tech_stack":"Unknown"}), x="tech_stack", color="sentiment", title="Sentiment by Tech Stack")
            st.plotly_chart(fig3, use_container_width=True)

    # Top Negative Keywords (same approach)
    st.markdown("### Top Negative Keywords")
    neg_text_series = merged[merged["sentiment"] == "negative"]["combined_text"].dropna()
    if len(neg_text_series) >= 3:
        try:
            vec_keywords = TfidfVectorizer(tokenizer=spacy_lemmatizer_tokenizer, max_features=50)
            X_keywords = vec_keywords.fit_transform(neg_text_series)
            sums = np.asarray(X_keywords.sum(axis=0)).ravel()
            vocab = np.array(vec_keywords.get_feature_names_out())
            kw_df = pd.DataFrame({"keyword": vocab, "score": sums}).sort_values("score", ascending=False).head(20)
            fig5 = px.bar(kw_df, x="keyword", y="score", title="Top Negative Keywords (TF-IDF)")
            st.plotly_chart(fig5, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate keywords: {e}")
    else:
        st.info("Not enough negative samples to extract keywords.")

# Live predictions (upload new recordings to predict)
if st.session_state.get("model_trained"):
    st.subheader(" Predict on Live (Unseen) Calls")
    st.success("Your custom model is trained! Upload new audio files here for live prediction.")

    live_audio_files = st.file_uploader("Upload new recordings for live prediction", type=["mp3","wav","m4a","aac"], accept_multiple_files=True, key="live_uploader")
    if live_audio_files:
            vectorizer, clf = st.session_state["sentiment_model"]
            try:
                whisper_model = load_whisper(whisper_size)
            except Exception as e:
                st.error(f"Error loading Whisper model: {e}")
                whisper_model = None

            results_rows = []

            if "live_cache" not in st.session_state:
                st.session_state["live_cache"] = {}  

            live_cache = st.session_state["live_cache"]

            for audio_file in live_audio_files:
                cid = os.path.splitext(os.path.basename(audio_file.name))[0]
                st.markdown("---")
                st.write(f"**File:** {audio_file.name}")

                try:
                    audio_bytes = audio_file.read()
                    audio_file.seek(0)

                    if cid not in live_cache:
                        with st.spinner("Transcribing..."):
                            raw_transcript = transcribe_with_whisper(audio_bytes, whisper_model, audio_file.name)
                        live_cache[cid] = raw_transcript
                    else:
                        raw_transcript = live_cache[cid]

                    st.write(f"**Transcript:** {raw_transcript}")
                    # ---------- 2) FORM: MANUAL INPUT + BUTTON (PERSISTENT) ----------
                    name_key = f"live_name_{cid}"
                    loc_key = f"live_loc_{cid}"
                    stack_key = f"live_stack_{cid}"

                    if name_key not in st.session_state:
                        st.session_state[name_key] = ""
                    if loc_key not in st.session_state:
                        st.session_state[loc_key] = ""
                    if stack_key not in st.session_state:
                        st.session_state[stack_key] = ""

                    with st.form(f"live_form_{cid}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            name_input = st.text_input(
                                "Student Name",
                                key=name_key,
                            )
                        with col2:
                            loc_input = st.text_input(
                                "Location",
                                key=loc_key,
                            )
                        with col3:
                            stack_input = st.text_input(
                                "Tech Stack / Course",
                                key=stack_key,
                            )

                        submitted = st.form_submit_button("Save & Predict")
                    if not submitted:
                        continue

                    # ---------- PREDICTION (HF OR CUSTOM) ----------
                    with st.spinner("Predicting..."):
                        if use_hf:
                            nlp_ob = load_hf_pipeline()
                            r = nlp_ob(raw_transcript[:4096])[0]
                            label = r["label"].lower()
                            if label == "positive":
                                prediction = "positive"
                            elif label == "negative":
                                prediction = "negative"
                            else:
                                prediction = "neutral"
                            score = float(r.get("score", 0))
                            model_used = "huggingface"
                        else:
                            prediction, score = predict_sentiment(vectorizer, clf, raw_transcript)
                            model_used = "custom_tf_idf"

                    # ---------- SAVE TO DB + UPDATE MERGED ----------
                    save_live_prediction_db(cid, raw_transcript, prediction, score)

                    final_name = st.session_state[name_key].strip() or None
                    final_loc = st.session_state[loc_key].strip() or None
                    final_stack = st.session_state[stack_key].strip() or None
                    row_for_log = {
                        "call_id": cid,
                        "student_name": final_name,
                        "tech_stack": final_stack,
                        "location": final_loc,
                        "remarks": None,
                        "transcript_text": raw_transcript,
                        "combined_text": raw_transcript,
                        "cleaned_text": " ".join(spacy_lemmatizer_tokenizer(raw_transcript)),
                        "label": None,
                        "sentiment": prediction,
                        "sentiment_score": float(score) if score is not None else None,
                    }
                    # store live row for this session's analytics
                    st.session_state["live_rows"].append(row_for_log)

                    try:
                        upsert_call_log(row_for_log)
                    except Exception as e:
                        st.warning(f"Could not upsert live call_log for {cid}: {e}")

                    results_rows.append({
                        "call_id": cid,
                        "transcript": raw_transcript,
                        "prediction": prediction,
                        "score": score,
                        "timestamp": datetime.now().isoformat(),
                    })

                    # ---------- 5) SHOW RESULT ----------
                    if prediction == "positive":
                        st.success(f"**Predicted Sentiment: {prediction.upper()}** (Confidence: {score:.2f})")
                    elif prediction == "negative":
                        st.error(f"**Predicted Sentiment: {prediction.upper()}** (Confidence: {score:.2f})")
                    else:
                        st.info(f"**Predicted Sentiment: {prediction.upper()}** (Confidence: {score:.2f})")

                except Exception as e:
                    st.error(f"Failed to process {audio_file.name}: {e}")
            # Export live predictions as CSV for user
            if results_rows:
                live_df = pd.DataFrame(results_rows)
                csv_buf = io.StringIO()
                live_df.to_csv(csv_buf, index=False)
                st.sidebar.download_button("Download live predictions CSV", data=csv_buf.getvalue(),
                                   file_name=f"live_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")
     
    else:
        st.info("Train a model and click the button to predict live.")

    # -----------------------------
    # Recommendations (rule + keyword heuristics)
    # -----------------------------
    st.subheader(" Recommendations")

    def gen_recos(df_: pd.DataFrame):
        recos = []
        total = len(df_)
        negc = (df_["sentiment"] == "negative").sum()
        if total > 0 and negc/total > 0.4:
            recos.append("Overall negative sentiment is high (>40%). Consider immediate coaching for counselors and revising scripts.")

        if "location" in df_.columns:
            by_loc = df_.groupby("location")["sentiment"].apply(lambda s: (s=="negative").mean()).sort_values(ascending=False)
            for loc, ratio in by_loc.items():
                if pd.notna(loc) and ratio >= 0.35:
                    recos.append(f"{loc}: Negative ratio {ratio:.0%}. Trial: reduce fees, add evening/weekend batches, or senior counselor callbacks.")

        if "tech_stack" in df_.columns:
            by_stack = df_.groupby("tech_stack")["sentiment"].apply(lambda s: (s=="negative").mean()).sort_values(ascending=False)
            for stack, ratio in by_stack.items():
                if pd.notna(stack) and ratio >= 0.35:
                    tips = {
                        "python": "emphasize job outcomes with case studies, add mini capstone demo",
                        "java": "offer installment plans, highlight placement partners",
                        "mern": "show live project repos and alumni testimonials",
                        "ai": "clarify math prerequisites and provide bridge modules"
                    }
                    extra = ""
                    k = str(stack).lower()
                    for key, val in tips.items():
                        if key in k:
                            extra = "; " + val
                            break
                    recos.append(f"{stack}: Negative ratio {ratio:.0%}. Address objections via FAQs{extra}.")

        text_all = " ".join(df_.get("combined_text", pd.Series(dtype=str)).dropna().astype(str).tolist()).lower()
        if any(k in text_all for k in ["fee", "fees", "price", "cost", "expensive"]):
            recos.append("Many fee-related objections → try scholarships, limited-time discounts, or EMI options.")
        if any(k in text_all for k in ["time", "timing", "slot", "schedule", "evening", "weekend"]):
            recos.append("Timing objections → add evening/weekend batches and flexible slots.")
        if any(k in text_all for k in ["location", "distance", "noida", "lucknow", "commute"]):
            recos.append("Location/commute issues → promote online/hybrid option and campus transfer flexibility.")
        if any(k in text_all for k in ["doubt", "support", "mentor", "teacher", "faculty"]):
            recos.append("Learning support concerns → advertise mentorship hours, doubt-solving sessions, and WhatsApp/Slack groups.")
        if any(k in text_all for k in ["job", "placement", "interview", "resume"]):
            recos.append("Career outcomes focus → showcase placement stats, resume/interview prep workshops.")
        return recos

    if merged is not None and isinstance(merged, pd.DataFrame) and len(merged) > 0:
        recommendations = gen_recos(merged)
    else:
        recommendations = []

    if recommendations:
        for r in recommendations:
            st.markdown(f"- {r}")
    else:
        st.info("No strong recommendations detected. With more data, insights will improve.")

# Export processed CSV
if 'merged' in locals() and merged is not None and save_intermediate:
    st.subheader(" Export Processed Data")
    out_csv = merged.copy().sort_values(by="call_id")
    out_buf = io.StringIO()
    out_csv.to_csv(out_buf, index=False)
    st.sidebar.download_button("Download processed CSV", data=out_buf.getvalue(),
                       file_name=f"processed_sentiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                       mime="text/csv")
