from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging

logger = logging.getLogger(__name__)

class LanguageDetectorService:
    """Service for detecting text language"""
    
    def __init__(self):
        # Set seed for consistent results
        DetectorFactory.seed = 0
        
    def detect(self, text: str) -> str:
        """
        Detect language of input text
        
        Args:
            text: Input text to detect language
            
        Returns:
            Language code (e.g., 'en', 'vi', 'fr')
        """
        try:
            # Clean text for better detection
            cleaned_text = self._clean_text(text)
            
            if len(cleaned_text.strip()) < 3:
                return "en"  # Default to English for very short texts
                
            detected_lang = detect(cleaned_text)
            logger.info(f"Detected language: {detected_lang} for text: {text[:50]}...")
            
            return detected_lang
            
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}. Defaulting to English.")
            return "en"  # Default to English if detection fails
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {e}")
            return "en"
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better language detection"""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove URLs, mentions, hashtags for better detection
        import re
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        return text.strip()
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return [
            'en', 'vi', 'fr', 'de', 'es', 'it', 'pt', 'ru', 'ja', 'ko', 
            'zh-cn', 'zh-tw', 'ar', 'hi', 'th', 'tr', 'pl', 'nl', 'sv'
        ]
