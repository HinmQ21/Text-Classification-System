import time
import logging
import re
import unicodedata
import numpy as np
from typing import Dict, Any, List, Optional, Union
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
        
        # Model configurations with display names
        self.model_configs = {
            "sentiment": {
                "models": {
                    "twitter-roberta": {
                        "full_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                        "display_name": "Roberta-base"
                    },
                    "twitter-roberta-xml": {
                        "full_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
                        "display_name": "XML-Roberta-base-multilingual"
                    }
                }
            },
            "spam": {
                "models": {
                    "roberta-sms": {
                        "full_name": "mariagrandury/roberta-base-finetuned-sms-spam-detection",
                        "display_name": "Roberta-base"
                    },
                    "distilbert-sms": {
                        "full_name": "mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection",
                        "display_name": "Distilbert-base"
                    }
                }
            },
            "topic": {
                "models": {
                    "bart-mnli": {
                        "full_name": "facebook/bart-large-mnli",
                        "display_name": "Bart-large"
                    },
                    "deberta-nli": {
                        "full_name": "tasksource/deberta-small-long-nli",
                        "display_name": "Deberta-small"
                    }
                }
            }
        }
        
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
            
            # Initialize sentiment analysis models
            self.models["sentiment"] = {}
            for model_key, model_info in self.model_configs["sentiment"]["models"].items():
                try:
                    self.models["sentiment"][model_key] = pipeline(
                        "sentiment-analysis",
                        model=model_info["full_name"],
                        return_all_scores=True
                    )
                    logger.info(f"Initialized sentiment model: {model_info['display_name']}")
                except Exception as e:
                    logger.error(f"Failed to initialize sentiment model {model_key}: {e}")
            
            # Initialize spam detection models
            self.models["spam"] = {}
            for model_key, model_info in self.model_configs["spam"]["models"].items():
                try:
                    self.models["spam"][model_key] = pipeline(
                        "text-classification",
                        model=model_info["full_name"],
                        return_all_scores=True
                    )
                    logger.info(f"Initialized spam model: {model_info['display_name']}")
                except Exception as e:
                    logger.error(f"Failed to initialize spam model {model_key}: {e}")
            
            # Initialize topic classification models
            self.models["topic"] = {}
            for model_key, model_info in self.model_configs["topic"]["models"].items():
                try:
                    self.models["topic"][model_key] = pipeline(
                        "zero-shot-classification",
                        model=model_info["full_name"]
                    )
                    logger.info(f"Initialized topic model: {model_info['display_name']}")
                except Exception as e:
                    logger.error(f"Failed to initialize topic model {model_key}: {e}")
            
            self.ready = True
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise e
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get available model configurations"""
        available = {}
        for task_type, config in self.model_configs.items():
            available[task_type] = {}
            for model_key, model_info in config["models"].items():
                # Check if model is actually loaded
                if task_type in self.models and model_key in self.models[task_type]:
                    available[task_type][model_key] = model_info["display_name"]
        return available

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
    
    def _ensemble_predictions(self, predictions: List[Dict[str, Any]], ensemble_method: str = "average") -> Dict[str, Any]:
        """
        Combine predictions from multiple models using ensemble methods
        
        Args:
            predictions: List of prediction dictionaries from individual models
            ensemble_method: Method for combining predictions ("average", "weighted", "voting")
            
        Returns:
            Combined prediction dictionary
        """
        if not predictions:
            raise ValueError("No predictions to ensemble")
        
        if len(predictions) == 1:
            return predictions[0]
        
        if ensemble_method == "average":
            # Average the probability scores
            all_labels = set()
            for pred in predictions:
                all_labels.update(pred["all_scores"].keys())
            
            combined_scores = {}
            for label in all_labels:
                scores = [pred["all_scores"].get(label, 0.0) for pred in predictions]
                combined_scores[label] = np.mean(scores)
            
            # Find the best prediction
            best_label = max(combined_scores.items(), key=lambda x: x[1])
            
            return {
                "label": best_label[0],
                "confidence": best_label[1],
                "all_scores": combined_scores
            }
        
        # Add other ensemble methods if needed
        else:
            logger.warning(f"Ensemble method '{ensemble_method}' not implemented, using average")
            return self._ensemble_predictions(predictions, "average")
        
    async def classify(self, text: str, model_type: str, language: str = None, temperature: float = 1.0, 
                      model_selection: Union[str, List[str]] = "all", enable_translation: bool = True) -> Dict[str, Any]:
        """
        Classify text with specified model type and parameters
        
        Args:
            text: Input text to classify
            model_type: Type of classification (sentiment, spam, topic)
            language: Language of input text (if None, will auto-detect)
            temperature: Temperature for softmax scaling (0.5-2.0)
            model_selection: Which models to use - "all", single model key, or list of model keys
            enable_translation: Whether to enable translation to English for non-English text
            
        Returns:
            Classification result with metadata
        """
        start_time = time.time()
        
        try:
            # Validate model type
            if model_type not in self.models or not self.models[model_type]:
                raise ValueError(f"Model type '{model_type}' not available or not initialized")
            
            # Get available models for this type
            available_models = list(self.models[model_type].keys())
            
            # Validate and process model selection
            if model_selection == "all":
                selected_models = available_models
            elif isinstance(model_selection, str):
                if model_selection in available_models:
                    selected_models = [model_selection]
                else:
                    raise ValueError(f"Model '{model_selection}' not available for type '{model_type}'")
            elif isinstance(model_selection, list):
                selected_models = [m for m in model_selection if m in available_models]
                if not selected_models:
                    raise ValueError(f"None of the specified models are available for type '{model_type}'")
            else:
                selected_models = available_models
            
            # Preprocess text
            processed_text = await self._preprocess_text(text, language, enable_translation)
            
            if not processed_text or processed_text.strip() == "":
                raise ValueError("Text preprocessing resulted in empty text")
                
            # Classify with selected models
            predictions = []
            model_results = {}
            
            for model_key in selected_models:
                try:
                    result = await self._model_classify(processed_text, model_type, model_key, temperature)
                    predictions.append(result)
                    model_results[model_key] = result
                    logger.info(f"Successfully classified with model {model_key}")
                except Exception as e:
                    logger.error(f"Error with model {model_key}: {e}")
                    continue
            
            if not predictions:
                raise ValueError("All models failed to classify the text")
            
            # Combine predictions if multiple models were used
            if len(predictions) > 1:
                final_result = self._ensemble_predictions(predictions, "average")
                is_ensemble = True
            else:
                final_result = predictions[0]
                is_ensemble = False
            
            processing_time = time.time() - start_time
            
            response = {
                "prediction": final_result["label"],
                "confidence": final_result["confidence"],
                "all_scores": final_result["all_scores"],
                "temperature": temperature,
                "processing_time": processing_time,
                "is_ensemble": is_ensemble,
                "models_used": selected_models,
                "individual_results": model_results if len(predictions) > 1 else None
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            processing_time = time.time() - start_time
            return {
                "prediction": "unknown",
                "confidence": 0.0,
                "all_scores": {},
                "temperature": temperature,
                "processing_time": processing_time,
                "is_ensemble": False,
                "models_used": [],
                "individual_results": None
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
    
    async def _preprocess_text(self, text: str, language: str = None, enable_translation: bool = True) -> str:
        """
        Advanced text preprocessing for real-world applications
        
        Args:
            text: Input text to preprocess
            language: Target language code (if None, will auto-detect)
            enable_translation: Whether to enable translation to English
            
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
            
            # Step 3: Translate to English if needed and enabled
            if enable_translation and language != "en" and language != "unknown":
                logger.info(f"Translating text from {language} to English...")
                processed_text = await self._translate_to_english(processed_text, language)
            elif not enable_translation and language != "en" and language != "unknown":
                logger.info(f"Translation disabled. Processing text in original language: {language}")
            
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
    
    async def _model_classify(self, text: str, model_type: str, model_key: str, temperature: float = 1.0) -> Dict[str, Any]:
        """Classify using actual ML models"""
        model = self.models[model_type][model_key]
        
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
    
    async def classify_batch(self, texts: List[str], model_type: str, batch_size: int,
                            language: str = None, temperature: float = 1.0,
                            model_selection: Union[str, List[str]] = "all", enable_translation: bool = True) -> List[Dict[str, Any]]:
        """
        Classify multiple texts using batch processing with HuggingFace pipelines

        Args:
            texts: List of input texts to classify
            model_type: Type of classification (sentiment, spam, topic)
            batch_size: Batch size for pipeline processing
            language: Language of the texts (if None, will auto-detect for each text)
            temperature: Temperature for softmax scaling (0.5-2.0)
            model_selection: Which models to use ("all", single model key, or list of model keys)
            enable_translation: Whether to enable translation to English for non-English texts

        Returns:
            List of dictionaries with prediction, confidence, all scores, and processing time for each text
        """
        start_time = time.time()

        try:
            if model_type not in self.models:
                raise ValueError(f"Model type '{model_type}' not supported")

            # Clamp temperature to valid range
            temperature = max(0.5, min(2.0, temperature))

            # Preprocess all texts
            processed_texts = []
            detected_languages = []

            for text in texts:
                processed_text = await self._preprocess_text(text, language, enable_translation)
                processed_texts.append(processed_text)

                # Detect language for each text if not provided
                if language is None:
                    detected_lang = self._detect_language(processed_text)
                    detected_languages.append(detected_lang)
                else:
                    detected_languages.append(language)

            # Determine which models to use
            available_models = list(self.models[model_type].keys())
            if not available_models:
                raise ValueError(f"No models available for type '{model_type}'")

            if model_selection == "all":
                selected_models = available_models
            elif isinstance(model_selection, str):
                if model_selection in available_models:
                    selected_models = [model_selection]
                else:
                    raise ValueError(f"Model '{model_selection}' not available for type '{model_type}'")
            elif isinstance(model_selection, list):
                selected_models = [m for m in model_selection if m in available_models]
                if not selected_models:
                    raise ValueError(f"None of the specified models are available for type '{model_type}'")
            else:
                selected_models = available_models

            # Perform batch classification with selected models
            all_predictions = []
            model_results = {}

            for model_key in selected_models:
                try:
                    batch_results = await self._model_classify_batch(
                        processed_texts, model_type, model_key, batch_size, temperature
                    )
                    all_predictions.append(batch_results)
                    model_results[model_key] = batch_results
                    logger.info(f"Successfully classified batch with model {model_key}")
                except Exception as e:
                    logger.error(f"Error with model {model_key}: {e}")
                    continue

            if not all_predictions:
                raise ValueError("All models failed to classify the texts")

            # Combine predictions if multiple models were used
            final_results = []
            for i in range(len(texts)):
                text_predictions = []
                text_model_results = {}

                for j, model_key in enumerate(selected_models):
                    if j < len(all_predictions) and i < len(all_predictions[j]):
                        text_predictions.append(all_predictions[j][i])
                        text_model_results[model_key] = all_predictions[j][i]

                if len(text_predictions) > 1:
                    final_result = self._ensemble_predictions(text_predictions, "average")
                    is_ensemble = True
                elif len(text_predictions) == 1:
                    final_result = text_predictions[0]
                    is_ensemble = False
                else:
                    # Fallback for failed predictions
                    final_result = {
                        "label": "unknown",
                        "confidence": 0.0,
                        "all_scores": {}
                    }
                    is_ensemble = False

                processing_time = time.time() - start_time

                result = {
                    "prediction": final_result["label"],
                    "confidence": final_result["confidence"],
                    "all_scores": final_result["all_scores"],
                    "temperature": temperature,
                    "processing_time": processing_time / len(texts),  # Average per text
                    "is_ensemble": is_ensemble,
                    "models_used": selected_models,
                    "individual_results": text_model_results if len(text_predictions) > 1 else None,
                    "detected_language": detected_languages[i]
                }

                final_results.append(result)

            return final_results

        except Exception as e:
            logger.error(f"Batch classification error: {e}")
            processing_time = time.time() - start_time

            # Return error results for all texts
            error_results = []
            for i in range(len(texts)):
                error_results.append({
                    "prediction": "unknown",
                    "confidence": 0.0,
                    "all_scores": {},
                    "temperature": temperature,
                    "processing_time": processing_time / len(texts),
                    "is_ensemble": False,
                    "models_used": [],
                    "individual_results": None,
                    "detected_language": "unknown"
                })

            return error_results

    async def _model_classify_batch(self, texts: List[str], model_type: str, model_key: str,
                                   batch_size: int, temperature: float = 1.0) -> List[Dict[str, Any]]:
        """Classify batch of texts using actual ML models with pipeline batch processing"""
        model = self.models[model_type][model_key]

        if model_type == "sentiment":
            # Use pipeline batch processing
            results = model(texts, batch_size=batch_size)

            batch_results = []
            for result in results:
                # Extract all scores and apply temperature
                all_scores = result  # List of {'label': str, 'score': float}
                labels = [item['label'].lower() for item in all_scores]
                probabilities = [item['score'] for item in all_scores]

                # Apply temperature scaling
                temp_scores = self._probabilities_to_logits_with_temperature(probabilities, temperature)

                # Create score dictionary
                score_dict = dict(zip(labels, temp_scores))

                # Find best prediction
                best_idx = np.argmax(temp_scores)
                best_label = labels[best_idx]
                best_confidence = temp_scores[best_idx]

                batch_results.append({
                    "label": best_label,
                    "confidence": best_confidence,
                    "all_scores": score_dict
                })

            return batch_results

        elif model_type == "spam":
            # Use pipeline batch processing
            results = model(texts, batch_size=batch_size)

            batch_results = []
            for result in results:
                all_scores = result

                # Map labels to more readable format
                mapped_probabilities = []
                mapped_labels = []
                for item in all_scores:
                    if item['label'] == "LABEL_1":
                        mapped_labels.append("spam")
                        mapped_probabilities.append(item['score'])
                    else:
                        mapped_labels.append("not_spam")
                        mapped_probabilities.append(item['score'])

                # Apply temperature scaling
                temp_scores = self._probabilities_to_logits_with_temperature(mapped_probabilities, temperature)

                # Create score dictionary
                score_dict = dict(zip(mapped_labels, temp_scores))

                # Find best prediction
                best_idx = np.argmax(temp_scores)
                best_label = mapped_labels[best_idx]
                best_confidence = temp_scores[best_idx]

                batch_results.append({
                    "label": best_label,
                    "confidence": best_confidence,
                    "all_scores": score_dict
                })

            return batch_results

        elif model_type == "topic":
            # Define candidate topics for zero-shot classification
            candidate_labels = [
                "technology", "sports", "politics", "entertainment",
                "business", "health", "education", "travel", "food", "science"
            ]

            batch_results = []
            # For zero-shot classification, we need to process each text individually
            # as the pipeline expects (text, candidate_labels) format
            for text in texts:
                result = model(text, candidate_labels)

                # Apply temperature scaling to the scores
                probabilities = result['scores']
                temp_scores = self._probabilities_to_logits_with_temperature(probabilities, temperature)

                # Create score dictionary
                score_dict = dict(zip(result['labels'], temp_scores))

                # Find best prediction
                best_idx = np.argmax(temp_scores)
                best_label = result['labels'][best_idx]
                best_confidence = temp_scores[best_idx]

                batch_results.append({
                    "label": best_label,
                    "confidence": best_confidence,
                    "all_scores": score_dict
                })

            return batch_results

    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self.ready
