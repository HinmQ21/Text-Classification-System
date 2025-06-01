import time
import logging
from typing import Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

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
                model="unitary/toxic-bert",
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
        
    async def classify(self, text: str, model_type: str, language: str = "en") -> Dict[str, Any]:
        """
        Classify text using specified model
        
        Args:
            text: Input text to classify
            model_type: Type of classification (sentiment, spam, topic)
            language: Language of the text
            
        Returns:
            Dictionary with prediction, confidence, and processing time
        """
        start_time = time.time()
        
        try:
            if model_type not in self.models:
                raise ValueError(f"Model type '{model_type}' not supported")
            
            # Translate to English if needed (simplified for demo)
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
    
    async def _preprocess_text(self, text: str, language: str) -> str:
        """Preprocess text before classification"""
        # Simple preprocessing
        text = text.strip()
        
        # For demo, we'll assume English or use simple translation
        if language != "en":
            # In a real implementation, you would use Google Gemini API here
            logger.info(f"Text is in {language}, would translate to English in production")
        
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
            label = "spam" if best_result['label'] == "TOXIC" else "not_spam"
            return {
                "label": label,
                "confidence": best_result['score']
            }
            
        elif model_type == "topic":
            # Define candidate topics for zero-shot classification
            candidate_labels = ["technology", "sports", "politics", "entertainment", "business", "health"]
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
            tech_words = ["computer", "software", "technology", "ai", "programming"]
            sports_words = ["football", "basketball", "soccer", "game", "team"]
            
            tech_count = sum(1 for word in tech_words if word in text_lower)
            sports_count = sum(1 for word in sports_words if word in text_lower)
            
            if tech_count > sports_count:
                return {"label": "technology", "confidence": 0.7}
            elif sports_count > 0:
                return {"label": "sports", "confidence": 0.7}
            else:
                return {"label": "general", "confidence": 0.5}
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self.ready
