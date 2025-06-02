import time
import logging
import re
import unicodedata
from typing import Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, DetectorFactory
import os

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

class TextClassifierService:
    """Service for text classification using pre-trained models"""
    
    def __init__(self):
        self.models = {}
        self.ready = False
        
    async def initialize(self):
        """Initialize pre-trained models"""
        try:
            logger.info("Initializing text classification models...")
            
            # Initialize sentiment analysis model
            self.models["sentiment"] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Initialize spam detection model (using a general classification model)
            self.models["spam"] = pipeline(
                "text-classification",
                model="mariagrandury/roberta-base-finetuned-sms-spam-detection",
                return_all_scores=True
            )
            
            # Initialize topic classification model
            self.models["topic"] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            self.ready = True
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            # Fallback to simple rule-based classification for demo
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize simple fallback models for demo purposes"""
        logger.info("Initializing fallback models...")
        self.models = {
            "sentiment": "fallback",
            "spam": "fallback", 
            "topic": "fallback"
        }
        self.ready = True
        
    async def classify(self, text: str, model_type: str, language: str = None) -> Dict[str, Any]:
        """
        Classify text using specified model
        
        Args:
            text: Input text to classify
            model_type: Type of classification (sentiment, spam, topic)
            language: Language of the text (if None, will auto-detect)
            
        Returns:
            Dictionary with prediction, confidence, and processing time
        """
        start_time = time.time()
        
        try:
            if model_type not in self.models:
                raise ValueError(f"Model type '{model_type}' not supported")
            
            # Advanced text preprocessing with language detection and translation
            processed_text = await self._preprocess_text(text, language)
            
            # Perform classification
            if self.models[model_type] == "fallback":
                result = self._fallback_classify(processed_text, model_type)
            else:
                result = await self._model_classify(processed_text, model_type)
            
            processing_time = time.time() - start_time
            
            return {
                "prediction": result["label"],
                "confidence": result["confidence"],
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            processing_time = time.time() - start_time
            return {
                "prediction": "unknown",
                "confidence": 0.0,
                "processing_time": processing_time
            }
    
    async def _preprocess_text(self, text: str, language: str = None) -> str:
        """
        Advanced text preprocessing for real-world applications
        
        Args:
            text: Input text to preprocess
            language: Target language code (if None, will auto-detect)
            
        Returns:
            Preprocessed text ready for classification
        """
        if not text or not text.strip():
            return ""
        
        try:
            # Step 1: Basic cleaning
            processed_text = self._clean_text(text)
            
            # Step 2: Detect language if not provided
            if language is None:
                language = self._detect_language(processed_text)
                logger.info(f"Auto-detected language: {language}")
            
            # Step 3: Final normalization
            processed_text = self._normalize_text(processed_text)
            
            # Note: Translation will be handled separately by Gemini API
            if language != "en" and language != "unknown":
                logger.info(f"Text is in {language}, translation will be handled by Gemini API")
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Text preprocessing error: {e}")
            # Return cleaned text even if other steps fail
            return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Handle social media mentions and hashtags (keep the word part)
        text = re.sub(r'@(\w+)', r'\1', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove excessive punctuation (keep normal punctuation)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove non-printable characters but keep emojis
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or ord(char) > 127)
        
        # Remove extra spaces and trim
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _detect_language(self, text: str) -> str:
        """Detect text language using langdetect"""
        try:
            # Only attempt detection if text is long enough
            if len(text.strip()) < 10:
                return "unknown"
            
            # Remove numbers and special characters for better detection
            clean_text = re.sub(r'[^\w\s]', ' ', text)
            clean_text = re.sub(r'\d+', '', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if len(clean_text) < 5:
                return "unknown"
            
            detected_lang = detect(clean_text)
            
            # Map some common language codes
            lang_mapping = {
                'vi': 'vi',  # Vietnamese
                'en': 'en',  # English
                'zh-cn': 'zh',  # Chinese
                'ja': 'ja',  # Japanese
                'ko': 'ko',  # Korean
                'th': 'th',  # Thai
                'fr': 'fr',  # French
                'de': 'de',  # German
                'es': 'es',  # Spanish
                'it': 'it',  # Italian
                'ru': 'ru',  # Russian
                'ar': 'ar',  # Arabic
            }
            
            return lang_mapping.get(detected_lang, detected_lang)
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown"
    
    def _normalize_text(self, text: str) -> str:
        """Final text normalization"""
        # Convert to lowercase for consistency (optional, depends on model requirements)
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure text is not empty
        if not text:
            return "empty text"
        
        # Limit text length (most models have token limits)
        max_length = 512  # Adjust based on your model's requirements
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary
            logger.info(f"Text truncated to {len(text)} characters")
        
        return text
    
    async def _model_classify(self, text: str, model_type: str) -> Dict[str, Any]:
        """Classify using actual ML models"""
        model = self.models[model_type]
        
        if model_type == "sentiment":
            results = model(text)
            # Get the highest confidence prediction
            best_result = max(results[0], key=lambda x: x['score'])
            return {
                "label": best_result['label'].lower(),
                "confidence": best_result['score']
            }
            
        elif model_type == "spam":
            results = model(text)
            best_result = max(results[0], key=lambda x: x['score'])
            # Map toxic-bert labels to spam/not_spam
            label = "spam" if best_result['label'] == "LABEL_1" else "not_spam"
            return {
                "label": label,
                "confidence": best_result['score']
            }
            
        elif model_type == "topic":
            # Define candidate topics for zero-shot classification
            candidate_labels = [
                "technology", "sports", "politics", "entertainment", 
                "business", "health", "education", "travel", "food", "science"
            ]
            result = model(text, candidate_labels)
            return {
                "label": result['labels'][0],
                "confidence": result['scores'][0]
            }
    
    def _fallback_classify(self, text: str, model_type: str) -> Dict[str, Any]:
        """Simple rule-based classification for demo"""
        text_lower = text.lower()
        
        if model_type == "sentiment":
            positive_words = ["good", "great", "excellent", "amazing", "love", "wonderful", "fantastic"]
            negative_words = ["bad", "terrible", "awful", "hate", "horrible", "worst"]
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return {"label": "positive", "confidence": 0.8}
            elif neg_count > pos_count:
                return {"label": "negative", "confidence": 0.8}
            else:
                return {"label": "neutral", "confidence": 0.6}
                
        elif model_type == "spam":
            spam_words = ["free", "win", "money", "click", "urgent", "limited", "offer"]
            spam_count = sum(1 for word in spam_words if word in text_lower)
            
            if spam_count >= 2:
                return {"label": "spam", "confidence": 0.7}
            else:
                return {"label": "not_spam", "confidence": 0.8}
                
        elif model_type == "topic":
            # Enhanced topic classification with more categories
            topic_keywords = {
                "technology": ["computer", "software", "technology", "ai", "programming", "digital", "internet", "app", "code", "algorithm", "data", "machine learning"],
                "sports": ["football", "basketball", "soccer", "game", "team", "player", "match", "championship", "score", "tournament", "athlete"],
                "business": ["company", "business", "market", "stock", "profit", "earnings", "revenue", "investment", "economy", "corporate", "financial"],
                "health": ["health", "medical", "doctor", "hospital", "medicine", "treatment", "patient", "disease", "exercise", "diet", "wellness"],
                "education": ["school", "university", "student", "teacher", "learning", "education", "study", "research", "academic", "knowledge"],
                "entertainment": ["movie", "music", "celebrity", "film", "concert", "show", "actor", "singer", "entertainment", "performance"],
                "politics": ["government", "president", "election", "policy", "politician", "vote", "democracy", "congress", "political"],
                "travel": ["travel", "vacation", "trip", "hotel", "flight", "tourist", "destination", "journey", "passport", "adventure"],
                "food": ["food", "restaurant", "recipe", "cooking", "chef", "meal", "cuisine", "ingredient", "dish", "taste"],
                "science": ["science", "research", "experiment", "discovery", "scientist", "laboratory", "theory", "study", "analysis"]
            }
            
            # Count keywords for each topic
            topic_scores = {}
            for topic, keywords in topic_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    topic_scores[topic] = score
            
            if topic_scores:
                # Get topic with highest score
                best_topic = max(topic_scores, key=topic_scores.get)
                confidence = min(0.9, 0.5 + (topic_scores[best_topic] * 0.1))
                return {"label": best_topic, "confidence": confidence}
            else:
                return {"label": "general", "confidence": 0.5}
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self.ready
