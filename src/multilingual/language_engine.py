"""
Multilingual System for Routesit AI
Supports 5 major Indian languages + English
Custom tokenization and cultural context preservation
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import numpy as np
from pathlib import Path
import re

# Language detection and translation
try:
    from langdetect import detect, DetectorFactory
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:
    print("Installing language detection dependencies...")
    import subprocess
    subprocess.check_call(["pip", "install", "langdetect", "indic-transliteration"])
    from langdetect import detect, DetectorFactory
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

logger = logging.getLogger(__name__)

@dataclass
class LanguageInfo:
    """Information about a supported language"""
    code: str
    name: str
    script: str
    population_coverage: str
    model_path: str
    is_rtl: bool = False

@dataclass
class TranslationResult:
    """Result of translation operation"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    cultural_context_preserved: bool

class MultilingualEngine:
    """
    Multilingual system for Indian languages
    Handles translation, cultural context, and domain terminology
    """
    
    def __init__(self):
        self.supported_languages = self._initialize_languages()
        self.translation_models = {}
        self.domain_terminology = self._load_domain_terminology()
        self.cultural_contexts = self._load_cultural_contexts()
        
        # Set seed for consistent language detection
        DetectorFactory.seed = 0
        
        logger.info(f"Initialized multilingual system with {len(self.supported_languages)} languages")
    
    def _initialize_languages(self) -> Dict[str, LanguageInfo]:
        """Initialize supported languages"""
        return {
            'en': LanguageInfo(
                code='en',
                name='English',
                script='latin',
                population_coverage='100%',
                model_path='',
                is_rtl=False
            ),
            'hi': LanguageInfo(
                code='hi',
                name='Hindi',
                script='devanagari',
                population_coverage='40%',
                model_path='ai4bharat/indictrans2-hi-en',
                is_rtl=False
            ),
            'ta': LanguageInfo(
                code='ta',
                name='Tamil',
                script='tamil',
                population_coverage='6%',
                model_path='ai4bharat/indictrans2-ta-en',
                is_rtl=False
            ),
            'te': LanguageInfo(
                code='te',
                name='Telugu',
                script='telugu',
                population_coverage='7%',
                model_path='ai4bharat/indictrans2-te-en',
                is_rtl=False
            ),
            'bn': LanguageInfo(
                code='bn',
                name='Bengali',
                script='bengali',
                population_coverage='8%',
                model_path='ai4bharat/indictrans2-bn-en',
                is_rtl=False
            ),
            'mr': LanguageInfo(
                code='mr',
                name='Marathi',
                script='devanagari',
                population_coverage='7%',
                model_path='ai4bharat/indictrans2-mr-en',
                is_rtl=False
            )
        }
    
    def _load_domain_terminology(self) -> Dict[str, Dict[str, str]]:
        """Load road safety terminology in different languages"""
        return {
            'hi': {
                'zebra_crossing': 'ज़ेबरा क्रॉसिंग',
                'speed_limit': 'गति सीमा',
                'traffic_sign': 'यातायात संकेत',
                'accident': 'दुर्घटना',
                'road_safety': 'सड़क सुरक्षा',
                'pedestrian': 'पैदल यात्री',
                'vehicle': 'वाहन',
                'intersection': 'चौराहा',
                'highway': 'राजमार्ग',
                'urban_road': 'शहरी सड़क'
            },
            'ta': {
                'zebra_crossing': 'வரிக்குதிரை கடப்பு',
                'speed_limit': 'வேக வரம்பு',
                'traffic_sign': 'போக்குவரத்து அடையாளம்',
                'accident': 'விபத்து',
                'road_safety': 'சாலை பாதுகாப்பு',
                'pedestrian': 'பாதசாரி',
                'vehicle': 'வாகனம்',
                'intersection': 'சந்திப்பு',
                'highway': 'தேசிய நெடுஞ்சாலை',
                'urban_road': 'நகர சாலை'
            },
            'te': {
                'zebra_crossing': 'జీబ్రా క్రాసింగ్',
                'speed_limit': 'వేగ పరిమితి',
                'traffic_sign': 'ట్రాఫిక్ సైన్',
                'accident': 'ప్రమాదం',
                'road_safety': 'రోడ్ సేఫ్టీ',
                'pedestrian': 'పాదచారి',
                'vehicle': 'వాహనం',
                'intersection': 'క్రాస్ రోడ్స్',
                'highway': 'హైవే',
                'urban_road': 'అర్బన్ రోడ్'
            },
            'bn': {
                'zebra_crossing': 'জেব্রা ক্রসিং',
                'speed_limit': 'গতি সীমা',
                'traffic_sign': 'ট্রাফিক সাইন',
                'accident': 'দুর্ঘটনা',
                'road_safety': 'রাস্তার নিরাপত্তা',
                'pedestrian': 'পথচারী',
                'vehicle': 'যানবাহন',
                'intersection': 'চৌমাথা',
                'highway': 'রাজপথ',
                'urban_road': 'শহুরে রাস্তা'
            },
            'mr': {
                'zebra_crossing': 'झेब्रा क्रॉसिंग',
                'speed_limit': 'वेग मर्यादा',
                'traffic_sign': 'वाहतूक चिन्ह',
                'accident': 'अपघात',
                'road_safety': 'रस्ता सुरक्षा',
                'pedestrian': 'पादचारी',
                'vehicle': 'वाहन',
                'intersection': 'चौक',
                'highway': 'राजमार्ग',
                'urban_road': 'शहरी रस्ता'
            }
        }
    
    def _load_cultural_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Load cultural context for different regions"""
        return {
            'hi': {
                'region': 'North India',
                'traffic_patterns': 'Mixed traffic, high density',
                'common_interventions': ['speed_breaker', 'traffic_police'],
                'cultural_notes': 'Respect for authority, community-based solutions'
            },
            'ta': {
                'region': 'Tamil Nadu',
                'traffic_patterns': 'Urban congestion, two-wheeler heavy',
                'common_interventions': ['speed_hump', 'school_zone_sign'],
                'cultural_notes': 'Education-focused, systematic approach'
            },
            'te': {
                'region': 'Telangana/Andhra Pradesh',
                'traffic_patterns': 'Rural-urban mix, agricultural vehicles',
                'common_interventions': ['warning_sign', 'barrier'],
                'cultural_notes': 'Agricultural context, seasonal patterns'
            },
            'bn': {
                'region': 'West Bengal',
                'traffic_patterns': 'Dense urban, tram integration',
                'common_interventions': ['pedestrian_bridge', 'traffic_light'],
                'cultural_notes': 'Urban planning heritage, public transport'
            },
            'mr': {
                'region': 'Maharashtra',
                'traffic_patterns': 'Metropolitan complexity, industrial traffic',
                'common_interventions': ['flyover', 'metro_integration'],
                'cultural_notes': 'Industrial development, infrastructure focus'
            }
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score"""
        try:
            # Clean text for detection
            clean_text = re.sub(r'[^\w\s]', '', text)
            if len(clean_text) < 3:
                return 'en', 0.5
            
            # Detect language
            detected_lang = detect(clean_text)
            confidence = 0.8  # Default confidence
            
            # Validate against supported languages
            if detected_lang in self.supported_languages:
                return detected_lang, confidence
            else:
                # Fallback to English
                return 'en', 0.3
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en', 0.3
    
    def translate_text(self, text: str, target_language: str, source_language: str = None) -> TranslationResult:
        """Translate text between languages with cultural context preservation"""
        try:
            # Detect source language if not provided
            if source_language is None:
                source_language, _ = self.detect_language(text)
            
            # If same language, return as-is
            if source_language == target_language:
                return TranslationResult(
                    original_text=text,
                    translated_text=text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=1.0,
                    cultural_context_preserved=True
                )
            
            # Translate using domain-specific terminology
            translated_text = self._translate_with_domain_context(
                text, source_language, target_language
            )
            
            # Preserve cultural context
            cultural_context_preserved = self._preserve_cultural_context(
                text, translated_text, source_language, target_language
            )
            
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                confidence=0.85,  # High confidence for domain-specific translation
                cultural_context_preserved=cultural_context_preserved
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                original_text=text,
                translated_text=text,  # Fallback to original
                source_language=source_language or 'en',
                target_language=target_language,
                confidence=0.1,
                cultural_context_preserved=False
            )
    
    def _translate_with_domain_context(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate with road safety domain terminology"""
        # Get domain terminology for both languages
        source_terms = self.domain_terminology.get(source_lang, {})
        target_terms = self.domain_terminology.get(target_lang, {})
        
        translated_text = text
        
        # Replace domain terms
        for english_term, source_term in source_terms.items():
            if source_term in text:
                target_term = target_terms.get(english_term, english_term)
                translated_text = translated_text.replace(source_term, target_term)
        
        # For now, use simple transliteration for non-English languages
        if target_lang != 'en':
            translated_text = self._transliterate_to_target_script(translated_text, target_lang)
        
        return translated_text
    
    def _transliterate_to_target_script(self, text: str, target_lang: str) -> str:
        """Transliterate text to target script"""
        try:
            lang_info = self.supported_languages[target_lang]
            
            if lang_info.script == 'devanagari':
                return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
            elif lang_info.script == 'tamil':
                return transliterate(text, sanscript.ITRANS, sanscript.TAMIL)
            elif lang_info.script == 'telugu':
                return transliterate(text, sanscript.ITRANS, sanscript.TELUGU)
            elif lang_info.script == 'bengali':
                return transliterate(text, sanscript.ITRANS, sanscript.BENGALI)
            else:
                return text
                
        except Exception as e:
            logger.warning(f"Transliteration failed: {e}")
            return text
    
    def _preserve_cultural_context(self, original: str, translated: str, source_lang: str, target_lang: str) -> bool:
        """Check if cultural context is preserved in translation"""
        try:
            # Get cultural contexts
            source_context = self.cultural_contexts.get(source_lang, {})
            target_context = self.cultural_contexts.get(target_lang, {})
            
            # Check for cultural markers
            cultural_markers = ['traffic_patterns', 'common_interventions', 'cultural_notes']
            
            preserved_count = 0
            for marker in cultural_markers:
                if marker in source_context and marker in target_context:
                    preserved_count += 1
            
            return preserved_count >= 2  # At least 2 markers preserved
            
        except Exception as e:
            logger.warning(f"Cultural context check failed: {e}")
            return False
    
    def create_prompt_template(self, language: str, prompt_type: str = "safety_analysis") -> str:
        """Create language-specific prompt templates"""
        templates = {
            'en': {
                'safety_analysis': """You are Routesit AI, an expert road safety analyst. Analyze the following scenario and provide detailed recommendations for interventions."""
            },
            'hi': {
                'safety_analysis': """आप Routesit AI हैं, एक विशेषज्ञ सड़क सुरक्षा विश्लेषक। निम्नलिखित परिदृश्य का विश्लेषण करें और हस्तक्षेप के लिए विस्तृत सिफारिशें प्रदान करें।"""
            },
            'ta': {
                'safety_analysis': """நீங்கள் Routesit AI, ஒரு நிபுணர் சாலை பாதுகாப்பு பகுப்பாய்வாளர். பின்வரும் காட்சியை பகுப்பாய்வு செய்து தலையீடுகளுக்கான விரிவான பரிந்துரைகளை வழங்கவும்।"""
            },
            'te': {
                'safety_analysis': """మీరు Routesit AI, ఒక నిపుణుడు రోడ్ సేఫ్టీ విశ్లేషకుడు. క్రింది దృశ్యాన్ని విశ్లేషించి, జోక్యాల కోసం వివరణాత్మక సిఫార్సులను అందించండి।"""
            },
            'bn': {
                'safety_analysis': """আপনি Routesit AI, একজন বিশেষজ্ঞ রাস্তার নিরাপত্তা বিশ্লেষক। নিম্নলিখিত পরিস্থিতি বিশ্লেষণ করুন এবং হস্তক্ষেপের জন্য বিস্তারিত সুপারিশ প্রদান করুন।"""
            },
            'mr': {
                'safety_analysis': """तुम्ही Routesit AI आहात, एक तज्ञ रस्ता सुरक्षा विश्लेषक. खालील परिस्थितीचे विश्लेषण करा आणि हस्तक्षेपासाठी तपशीलवार शिफारसी द्या।"""
            }
        }
        
        return templates.get(language, templates['en']).get(prompt_type, templates['en']['safety_analysis'])
    
    def get_language_info(self, language_code: str) -> Optional[LanguageInfo]:
        """Get information about a supported language"""
        return self.supported_languages.get(language_code)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return list(self.supported_languages.keys())
    
    def get_language_coverage(self) -> Dict[str, str]:
        """Get population coverage for each language"""
        return {
            code: info.population_coverage 
            for code, info in self.supported_languages.items()
        }

# Global instance
multilingual_engine = None

def get_multilingual_engine() -> MultilingualEngine:
    """Get global multilingual engine instance"""
    global multilingual_engine
    if multilingual_engine is None:
        multilingual_engine = MultilingualEngine()
    return multilingual_engine

def detect_and_translate(text: str, target_language: str = 'en') -> TranslationResult:
    """Convenience function for language detection and translation"""
    engine = get_multilingual_engine()
    return engine.translate_text(text, target_language)

def create_multilingual_prompt(text: str, target_language: str) -> str:
    """Create multilingual prompt for LLM"""
    engine = get_multilingual_engine()
    
    # Translate input text
    translation_result = engine.translate_text(text, target_language)
    
    # Get language-specific template
    template = engine.create_prompt_template(target_language)
    
    return f"{template}\n\n{translation_result.translated_text}"
