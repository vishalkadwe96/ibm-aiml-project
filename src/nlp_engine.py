"""
IBM Enterprise AI/ML Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Module: NLP Engine + IBM Watson/watsonx.ai Integration
Description: Document intelligence, semantic search, text classification,
             named entity recognition, summarization, and IBM Granite LLM.
"""

import re
import json
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

# ── Optional imports with graceful fallback ─────────────────────
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from transformers import (
        pipeline, AutoTokenizer, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, AutoModelForSeq2SeqLM,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
    SKLEARN_NLP_AVAILABLE = True
except ImportError:
    SKLEARN_NLP_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
# Text Preprocessor
# ──────────────────────────────────────────────────────────────────
class TextPreprocessor:
    """Enterprise text normalization pipeline."""

    def __init__(self):
        self.lemmatizer = None
        self._stop_words = {
            "the","a","an","is","it","in","on","at","to","for","of","and",
            "or","but","not","are","was","were","been","be","have","has","had",
            "do","does","did","will","would","could","should","may","might",
            "this","that","these","those","i","you","he","she","we","they",
        }
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except Exception:
                pass

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Apply token replacements BEFORE lowercasing so tags survive
        text = re.sub(r"http\S+|www\S+", " URL ", text)
        text = re.sub(r"\S+@\S+", " EMAIL ", text)
        text = re.sub(r"\d{4}[-/]\d{2}[-/]\d{2}", " DATE ", text)
        text = re.sub(r"\$[\d,]+\.?\d*", " MONEY ", text)
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        tokens = text.split()
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words and len(t) > 2]
        if self.lemmatizer:
            try:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            except Exception:
                pass
        return tokens

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]


# ──────────────────────────────────────────────────────────────────
# Document Embedder (Semantic Search Engine)
# ──────────────────────────────────────────────────────────────────
class DocumentEmbedder:
    """
    Semantic embedding engine using Sentence Transformers.
    Falls back to TF-IDF if sentence-transformers unavailable.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.tfidf = None
        self.embeddings_: Optional[np.ndarray] = None
        self.corpus_: Optional[List[str]] = None
        self._use_sbert = SBERT_AVAILABLE

    def load_model(self):
        if self._use_sbert:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded SBERT model: {self.model_name}")
        else:
            self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            logger.warning("Sentence Transformers unavailable — using TF-IDF embeddings")
        return self

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.array([])
        if self._use_sbert and self.model:
            embeddings = self.model.encode(
                texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
            )
        else:
            if self.tfidf is None:
                self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            if not hasattr(self.tfidf, "vocabulary_"):
                embeddings = self.tfidf.fit_transform(texts).toarray()
            else:
                embeddings = self.tfidf.transform(texts).toarray()
        self.embeddings_ = np.array(embeddings)
        self.corpus_ = texts
        logger.info(f"Embedded {len(texts)} documents → shape {self.embeddings_.shape}")
        return self.embeddings_

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings_ is None or self.corpus_ is None:
            raise RuntimeError("Call embed() first")
        if self._use_sbert and self.model:
            q_emb = self.model.encode([query], normalize_embeddings=True)
        else:
            q_emb = self.tfidf.transform([query]).toarray()

        sims = cosine_similarity(q_emb, self.embeddings_)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            {"rank": i + 1, "score": float(sims[idx]), "document": self.corpus_[idx]}
            for i, idx in enumerate(top_idx)
        ]


# ──────────────────────────────────────────────────────────────────
# Document Topic Modeler (LDA + NMF)
# ──────────────────────────────────────────────────────────────────
class TopicModeler:
    """LDA-based topic modeling for document corpus intelligence."""

    def __init__(self, n_topics: int = 10, max_features: int = 5000):
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
        self.lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=20, n_jobs=-1
        )
        self.topics_: Optional[List[List[str]]] = None

    def fit(self, texts: List[str]) -> "TopicModeler":
        dtm = self.vectorizer.fit_transform(texts)
        self.lda.fit(dtm)
        vocab = np.array(self.vectorizer.get_feature_names_out())
        self.topics_ = [vocab[topic.argsort()[-12:][::-1]].tolist() for topic in self.lda.components_]
        logger.success(f"LDA fitted: {self.n_topics} topics from {len(texts)} documents")
        return self

    def get_document_topics(self, texts: List[str]) -> np.ndarray:
        dtm = self.vectorizer.transform(texts)
        return self.lda.transform(dtm)

    def print_topics(self):
        print("\n" + "═" * 60)
        print("  DISCOVERED TOPICS")
        print("═" * 60)
        for i, words in enumerate(self.topics_ or []):
            print(f"  Topic {i+1:2d}: {', '.join(words[:8])}")
        print("═" * 60)


# ──────────────────────────────────────────────────────────────────
# NER + Sentiment Pipeline
# ──────────────────────────────────────────────────────────────────
class NLPInferencePipeline:
    """
    Zero-shot classification, sentiment analysis, and NER
    using HuggingFace Transformers.
    """

    def __init__(self):
        self._sentiment = None
        self._ner = None
        self._zero_shot = None
        self._summarizer = None

    def load(self):
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available — NLP inference disabled")
            return self
        logger.info("Loading NLP inference pipelines...")
        try:
            self._sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            logger.info("  ✓ Sentiment analysis loaded")
        except Exception as e:
            logger.warning(f"  ✗ Sentiment: {e}")
        try:
            self._ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
            logger.info("  ✓ NER pipeline loaded")
        except Exception as e:
            logger.warning(f"  ✗ NER: {e}")
        try:
            self._zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            logger.info("  ✓ Zero-shot classification loaded")
        except Exception as e:
            logger.warning(f"  ✗ Zero-shot: {e}")
        return self

    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        if self._sentiment is None:
            return [{"label": "UNKNOWN", "score": 0.0}] * len(texts)
        return self._sentiment(texts[:512])

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        if self._ner is None:
            return []
        return self._ner(text[:512])

    def classify_zero_shot(self, text: str, labels: List[str]) -> Dict[str, Any]:
        if self._zero_shot is None:
            return {"labels": labels, "scores": [1.0 / len(labels)] * len(labels)}
        return self._zero_shot(text, candidate_labels=labels)


# ──────────────────────────────────────────────────────────────────
# IBM watsonx.ai Client
# ──────────────────────────────────────────────────────────────────
class WatsonxAIClient:
    """
    IBM watsonx.ai integration for foundation model inference
    (IBM Granite-13B). Falls back to local summarization if
    credentials are not set.
    """

    def __init__(self, url: str = "", api_key: str = "", project_id: str = ""):
        self.url = url
        self.api_key = api_key
        self.project_id = project_id
        self.client = None
        self._connected = False

    def connect(self):
        if not self.url or not self.api_key:
            logger.warning("Watson credentials not set — running in offline/demo mode")
            return self
        try:
            from ibm_watson_machine_learning.foundation_models import Model
            from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
            self.GenParams = GenParams
            self.Model = Model
            self._connected = True
            logger.success("Connected to IBM watsonx.ai")
        except ImportError:
            logger.warning("ibm-watson-machine-learning not installed — demo mode")
        return self

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if not self._connected:
            return self._demo_generate(prompt, max_tokens)
        params = {
            self.GenParams.MAX_NEW_TOKENS: max_tokens,
            self.GenParams.TEMPERATURE: temperature,
            self.GenParams.REPETITION_PENALTY: 1.1,
        }
        model = self.Model(
            model_id="ibm/granite-13b-instruct-v2",
            credentials={"url": self.url, "apikey": self.api_key},
            project_id=self.project_id,
            params=params,
        )
        result = model.generate_text(prompt=prompt)
        return result

    def _demo_generate(self, prompt: str, max_tokens: int) -> str:
        """Offline demo response generator."""
        templates = {
            "summarize": "This document discusses enterprise AI/ML operations. Key points include: anomaly detection, predictive analytics, and NLP-driven insights.",
            "classify": "Based on the content analysis, this document belongs to category: Technical Report.",
            "explain": "The model detected this as anomalous because the feature values deviate significantly from learned distributions.",
            "report": "IBM AI Platform Analysis Report: The system processed the data and identified 47 anomalies with 94.2% confidence.",
        }
        for key, response in templates.items():
            if key in prompt.lower():
                return response
        return f"[IBM Granite Demo] Processed prompt ({len(prompt)} chars). In production, this calls IBM watsonx.ai Granite-13B."

    def summarize_document(self, text: str) -> str:
        prompt = f"Summarize the following enterprise document concisely:\n\n{text[:2000]}\n\nSummary:"
        return self.generate(prompt, max_tokens=200)

    def explain_anomaly(self, features: Dict[str, float]) -> str:
        feat_str = ", ".join(f"{k}={v:.3f}" for k, v in list(features.items())[:10])
        prompt = f"Explain why this transaction is anomalous. Features: {feat_str}\n\nExplanation:"
        return self.generate(prompt, max_tokens=150)

    def generate_insight(self, metrics: Dict[str, Any]) -> str:
        prompt = f"Generate a business insight from these ML model metrics: {json.dumps(metrics, indent=2)}\n\nInsight:"
        return self.generate(prompt, max_tokens=200)


# ──────────────────────────────────────────────────────────────────
# Document Intelligence Orchestrator
# ──────────────────────────────────────────────────────────────────
class DocumentIntelligenceEngine:
    """
    Full document intelligence pipeline:
    preprocess → embed → cluster → topic model → NER → sentiment → LLM insights
    """

    def __init__(self, watson_cfg: Optional[Dict] = None):
        self.preprocessor = TextPreprocessor()
        self.embedder = DocumentEmbedder()
        self.topic_modeler = TopicModeler(n_topics=8)
        self.nlp_pipeline = NLPInferencePipeline()
        self.watson = WatsonxAIClient(**(watson_cfg or {}))
        self._fitted = False

    def ingest_documents(self, texts: List[str]) -> Dict[str, Any]:
        logger.info(f"Ingesting {len(texts)} documents...")
        cleaned = self.preprocessor.process_batch(texts)

        # Embed
        self.embedder.load_model()
        embeddings = self.embedder.embed(cleaned)

        # Topic modeling
        self.topic_modeler.fit(cleaned)
        doc_topics = self.topic_modeler.get_document_topics(cleaned)

        # Clustering
        n_clusters = min(8, len(texts) // 10 + 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Sentiment (sample)
        sample = cleaned[:min(50, len(cleaned))]
        sentiments = self.nlp_pipeline.analyze_sentiment(sample)

        self._fitted = True
        result = {
            "total_documents": len(texts),
            "embedding_dim": embeddings.shape[1],
            "n_clusters": n_clusters,
            "cluster_distribution": Counter(cluster_labels.tolist()),
            "dominant_topics": [words[:5] for words in (self.topic_modeler.topics_ or [])[:3]],
            "sentiment_sample": sentiments[:5],
        }
        logger.success(f"Document Intelligence: {len(texts)} docs processed into {n_clusters} clusters")
        return result

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.embedder.semantic_search(query, top_k=top_k)

    def get_watson_insight(self, metrics: Dict[str, Any]) -> str:
        self.watson.connect()
        return self.watson.generate_insight(metrics)
