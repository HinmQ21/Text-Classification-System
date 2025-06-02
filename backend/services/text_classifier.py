import time
import logging
import re
import unicodedata
import numpy as np
from typing import Dict, Any, List
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, DetectorFactory
import os
import google.generativeai as genai

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

class TextClassifierService:
    """Service for text classification using pre-trained models"""
    
    def __init__(self):
        self.models = {}
        self.ready = False
        self.gemini_model = None
        self._initialize_gemini()
        
    def _initialize_gemini(self):
        """Initialize Gemini API for translation"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                logger.warning("GEMINI_API_KEY not found in environment variables. Translation will be disabled.")
                return
            
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
            logger.info("Gemini API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            self.gemini_model = None
        
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
            # Remove fallback initialization since it's not needed
            raise e
    
    def _softmax_with_temperature(self, logits: List[float], temperature: float = 1.0) -> List[float]:
        """
        Apply softmax with temperature scaling
        
        Args:
            logits: Raw prediction scores (logits, not probabilities)
            temperature: Temperature parameter (0.5-2.0)
                        - Lower values (< 1.0) make the model more confident
                        - Higher values (> 1.0) make the model less confident
                        
        Returns:
            Probability distribution after temperature scaling
        """
        if temperature <= 0:
            temperature = 1.0
            
        # Convert to numpy array for easier computation
        logits_array = np.array(logits, dtype=np.float64)
        
        # Apply temperature scaling
        scaled_logits = logits_array / temperature
        
        # Apply softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # Subtract max for numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        return probabilities.tolist()
    
    def _probabilities_to_logits_with_temperature(self, probabilities: List[float], temperature: float = 1.0) -> List[float]:
        """
        Convert probabilities to logits and apply temperature scaling
        
        Args:
            probabilities: Probability distribution from model output
            temperature: Temperature parameter (0.5-2.0)
                        
        Returns:
            New probability distribution after temperature scaling
        """
        if temperature <= 0:
            temperature = 1.0
            
        # Convert to numpy array
        probs_array = np.array(probabilities, dtype=np.float64)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        probs_array = np.clip(probs_array, epsilon, 1.0 - epsilon)
        
        # Convert probabilities to logits
        logits = np.log(probs_array)
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Apply softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        new_probabilities = exp_logits / np.sum(exp_logits)
        
        return new_probabilities.tolist()
        
    async def classify(self, text: str, model_type: str, language: str = None, temperature: float = 1.0) -> Dict[str, Any]:
        """
        Classify text using specified model
        
        Args:
            text: Input text to classify
            model_type: Type of classification (sentiment, spam, topic)
            language: Language of the text (if None, will auto-detect)
            temperature: Temperature for softmax scaling (0.5-2.0)
            
        Returns:
            Dictionary with prediction, confidence, all scores, and processing time
        """
        start_time = time.time()
        
        try:
            if model_type not in self.models:
                raise ValueError(f"Model type '{model_type}' not supported")
            
            # Clamp temperature to valid range
            temperature = max(0.5, min(2.0, temperature))
            
            # Advanced text preprocessing with language detection and translation
            processed_text = await self._preprocess_text(text, language)
            
            # Perform classification
            result = await self._model_classify(processed_text, model_type, temperature)
            
            processing_time = time.time() - start_time
            
            return {
                "prediction": result["label"],
                "confidence": result["confidence"],
                "all_scores": result["all_scores"],
                "temperature": temperature,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            processing_time = time.time() - start_time
            return {
                "prediction": "unknown",
                "confidence": 0.0,
                "all_scores": [],
                "temperature": temperature,
                "processing_time": processing_time
            }
    
    async def _translate_to_english(self, text: str, source_language: str) -> str:
        """
        Translate text to English using Gemini API
        
        Args:
            text: Text to translate
            source_language: Source language code
            
        Returns:
            Translated text in English
        """
        if not self.gemini_model:
            logger.warning("Gemini API not available. Returning original text.")
            return text
        
        if source_language == "en" or source_language == "unknown":
            return text
        
        try:
            # Create translation prompt
            prompt = f"""
You are a professional translator.

Your task:
1. Translate the input text into English.
2. If the provided source language '{source_language}' does NOT match the actual language of the input, automatically detect the correct language and translate accordingly.
3. Return ONLY the translated version of the input text, with no explanations, notes, or formatting.
4. Keep the output as a single text block that mirrors the structure of the input.

Input text:
{text}
"""
            
            response = self.gemini_model.generate_content(prompt)
            translated_text = response.text.strip()
            
            # Validate translation
            if translated_text and len(translated_text) > 0:
                logger.info(f"Successfully translated text from {source_language} to English")
                return translated_text
            else:
                logger.warning("Gemini returned empty translation. Using original text.")
                return text
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
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
            
            # Step 3: Translate to English if needed
            if language != "en" and language != "unknown":
                logger.info(f"Translating text from {language} to English...")
                processed_text = await self._translate_to_english(processed_text, language)
            
            # Step 4: Final normalization
            processed_text = self._normalize_text(processed_text)
            
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
    
    async def _model_classify(self, text: str, model_type: str, temperature: float = 1.0) -> Dict[str, Any]:
        """Classify using actual ML models"""
        model = self.models[model_type]
        
        if model_type == "sentiment":
            results = model(text)
            # Extract all scores and apply temperature
            all_scores = results[0]  # List of {'label': str, 'score': float}
            labels = [item['label'].lower() for item in all_scores]
            probabilities = [item['score'] for item in all_scores]  # These are already probabilities
            
            # Apply temperature scaling (convert probabilities to logits first)
            temp_scores = self._probabilities_to_logits_with_temperature(probabilities, temperature)
            
            # Create score dictionary
            score_dict = dict(zip(labels, temp_scores))
            
            # Find best prediction
            best_idx = np.argmax(temp_scores)
            best_label = labels[best_idx]
            best_confidence = temp_scores[best_idx]
            
            return {
                "label": best_label,
                "confidence": best_confidence,
                "all_scores": score_dict
            }
            
        elif model_type == "spam":
            results = model(text)
            all_scores = results[0]
            
            # Map labels to more readable format
            mapped_probabilities = []
            mapped_labels = []
            for item in all_scores:
                if item['label'] == "LABEL_1":
                    mapped_labels.append("spam")
                    mapped_probabilities.append(item['score'])  # These are probabilities
                else:
                    mapped_labels.append("not_spam")
                    mapped_probabilities.append(item['score'])  # These are probabilities
            
            # Apply temperature scaling (convert probabilities to logits first)
            temp_scores = self._probabilities_to_logits_with_temperature(mapped_probabilities, temperature)
            
            # Create score dictionary
            score_dict = dict(zip(mapped_labels, temp_scores))
            
            # Find best prediction
            best_idx = np.argmax(temp_scores)
            best_label = mapped_labels[best_idx]
            best_confidence = temp_scores[best_idx]
            
            return {
                "label": best_label,
                "confidence": best_confidence,
                "all_scores": score_dict
            }
            
        elif model_type == "topic":
            # Define candidate topics for zero-shot classification
            candidate_labels = [
                "technology", "sports", "politics", "entertainment", 
                "business", "health", "education", "travel", "food", "science"
            ]
            result = model(text, candidate_labels)
            
            # Apply temperature scaling to the scores (these are probabilities)
            probabilities = result['scores']
            temp_scores = self._probabilities_to_logits_with_temperature(probabilities, temperature)
            
            # Create score dictionary
            score_dict = dict(zip(result['labels'], temp_scores))
            
            # Find best prediction
            best_idx = np.argmax(temp_scores)
            best_label = result['labels'][best_idx]
            best_confidence = temp_scores[best_idx]
            
            return {
                "label": best_label,
                "confidence": best_confidence,
                "all_scores": score_dict
            }
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self.ready
