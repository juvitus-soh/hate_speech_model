"""
Cameroon Hate Speech Detection System

A comprehensive hate speech detection system specifically designed for Cameroonian social media content.
Uses keyword-triggered AI analysis for optimal performance and accuracy.

Features:
- 160+ Cameroon-specific hate speech keywords
- Multi-language support (French, English, Pidgin)
- Keyword-triggered AI analysis for efficiency
- Pre-trained transformer models
- Real-time processing capabilities
"""

import re
import torch
from transformers import pipeline
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HateSpeechResult:
    """Data class for hate speech detection results"""
    text: str
    is_hate_speech: bool
    confidence: float
    detected_keywords: List[str]
    category: str
    severity: str
    timestamp: datetime
    explanation: str

class CameroonKeywordsDetector:
    """Comprehensive keyword detector for Cameroonian hate speech terms"""

    def __init__(self):
        """Initialize with extensive Cameroon-specific keyword database"""
        self.keywords = {
            # Ethnic and tribal hatred
            'ethnic_hatred': {
                'terms': [
                    'tontinards', 'sardinards', 'graffi', 'graffeurs', 'nges man', 'bamendaman',
                    'kumba boys', 'bamilekes', 'betis', 'ewondos', 'hausas du nord', 'kirdi',
                    'moutons du nord', 'nordistes', 'sudistes', 'anglophones ennemis',
                    'francophones colonisateurs', 'separatistes', 'federalistes terroristes',
                    'biafrais', 'nigerians illegaux', 'tchadiens envahisseurs', 'centrafricains refugies',
                    'come no go', 'anglo dogs', 'frog speakers', 'french slaves', 'colonial masters',
                    'beti-bulu', 'fang-beti', 'bassa-bakoko', 'douala-littoral', 'bafia-eton'
                ],
                'severity': 'high',
                'category': 'ethnic_hatred'
            },

            # Linguistic discrimination
            'linguistic_discrimination': {
                'terms': [
                    'anglofous', 'anglofolles', 'francofous', 'francofolles', 'pidgin speakers',
                    'broken english', 'french colonized', 'anglophone rebels', 'francophone dictators',
                    'english dogs', 'camerounais authentiques', 'vrais camerounais', 'faux patriotes',
                    'separatist anglos', 'terrorist anglophones', 'secessionist pigs', 'amba terrorists',
                    'restoration terrorists', 'ghost town supporters', 'ghost town criminals'
                ],
                'severity': 'high',
                'category': 'linguistic_discrimination'
            },

            # Religious hatred
            'religious_hatred': {
                'terms': [
                    'boko haram supporters', 'muslim terrorists', 'christian crusaders', 'kafirs',
                    'infidels camerounais', 'islamic invaders', 'jihadists nordistes', 'animist primitives',
                    'feticheurs', 'sorciers', 'traditionalist backwards', 'church exploiters',
                    'mosques terrorists', 'halal killers', 'sunday hypocrites', 'ramadan criminals'
                ],
                'severity': 'high',
                'category': 'religious_hatred'
            },

            # Political repression and dissent
            'political_dissent': {
                'terms': [
                    'paul biya dictateur', 'biya must go', 'down with cpdm', 'corrupt cpdm',
                    'rdpc voleurs', 'cpdm criminals', 'regime criminelle', 'dictature camerounaise',
                    'election truquee', 'fraud electoral', 'rigged elections', 'fake results',
                    'stolen votes', 'ballot stuffing', 'vote buying', 'constitutional council puppets',
                    'elecam corrupt', 'observers banned', 'international conspiracy', 'neo-colonial plot',
                    'kamtoistes terroristes', 'mrc criminals', 'opposition terrorists', 'enemies of state',
                    'western puppets', 'regime change agents', 'democracy destroyers', 'chaos agents'
                ],
                'severity': 'medium',
                'category': 'political_dissent'
            },

            # Incitement to violence
            'violence_incitement': {
                'terms': [
                    'kill them all', 'elimination physique', 'nettoyage ethnique', 'guerre civile',
                    'civil war necessary', 'armed revolution', 'violence solution', 'burn everything',
                    'destroy anglophone regions', 'crush separatists', 'military solution final',
                    'shoot protesters', 'arrest opposition', 'prison for critics', 'torture enemies',
                    'kadija must pay', 'maurice kamto prison', 'joshua osih elimination', 'cabral libii arrest',
                    'akere muna traitor', 'ni john fru ndi enemy', 'sisiku ayuk tabe terrorist'
                ],
                'severity': 'high',
                'category': 'violence_incitement'
            },

            # Regional discrimination
            'regional_discrimination': {
                'terms': [
                    'wadjo', 'wouri', 'bamenda grassfield', 'kumba boys', 'buea students',
                    'douala gangsters', 'yaounde bureaucrats', 'maroua backwards', 'ngaoundere cattle',
                    'garoua corrupt', 'bamenda rebels', 'limbe fishermen', 'kribi lazy', 'edea drunkards',
                    'bafoussam merchants', 'foumban traditional', 'tibati primitive', 'yokadouma forest',
                    'bertoua diamonds', 'sangmelima cocoa', 'ebolowa backward', 'ambam border criminals'
                ],
                'severity': 'medium',
                'category': 'regional_discrimination'
            },

            # Gender-based discrimination and hate
            'gender_discrimination': {
                'terms': [
                    'femmes politicians incompetent', 'women weak leaders', 'girls only marriage',
                    'feminist destroyers', 'gender equality western', 'traditional women only',
                    'modern women prostitutes', 'working women neglectful', 'single mothers sinful',
                    'divorced women failures', 'widows cursed', 'barren women punished', 'lgbt abomination',
                    'homosexual western disease', 'gay marriage illegal', 'lesbian criminal', 'transgender mentally ill'
                ],
                'severity': 'high',
                'category': 'gender_discrimination'
            },

            # Election-specific hate speech (2025 context)
            'election_hatred': {
                'terms': [
                    'election wahala', 'vote dem don buy', 'rigged again', 'same results always',
                    'international observers waste', 'opposition jailed', 'campaign restricted', 'media controlled',
                    'ballot boxes stuffed', 'results predetermined', 'constitutional council biased', 'elecam corrupt officials',
                    'diaspora votes cancelled', 'registration blocked', 'voter cards missing', 'polling stations closed',
                    'military intimidation', 'violence during voting', 'protesters arrested', 'opposition eliminated',
                    'fake news spreading', 'social media blocked', 'internet shutdown', 'communications cut'
                ],
                'severity': 'medium',
                'category': 'election_hatred'
            },

            # Cameroonian Pidgin specific terms
            'pidgin_hate': {
                'terms': [
                    'dem nor fit', 'yi nor sabi', 'wuna nor get sense', 'dem na animal',
                    'yi na dog', 'wuna na goat', 'dem na pig', 'yi nor fine',
                    'ugly face dem get', 'nonsense people', 'foolish tribe', 'stupid region',
                    'useless government', 'nonsense election', 'stupid voters', 'foolish democracy',
                    'we nor go gree', 'dem don die finish', 'e go hot', 'wahala go dey',
                    'trouble go start', 'fight go begin', 'palava go full ground', 'kasala go burst'
                ],
                'severity': 'medium',
                'category': 'pidgin_hate'
            },

            # Anti-government and institutional hatred
            'institutional_hatred': {
                'terms': [
                    'corrupt ministers', 'thief governors', 'criminal prefects', 'useless parliamentarians',
                    'fake senators', 'puppet mayors', 'bought judges', 'corrupt police', 'military oppressors',
                    'gendarmerie criminals', 'bir terrorists', 'administrative corrupt', 'civil servants lazy',
                    'public servants thieves', 'tax collectors corrupt', 'customs criminals', 'immigration corrupt',
                    'education minister incompetent', 'health minister killer', 'finance minister thief',
                    'defense minister oppressor', 'interior minister dictator', 'justice minister corrupt'
                ],
                'severity': 'medium',
                'category': 'institutional_hatred'
            },

            # Conspiracy theories and misinformation
            'conspiracy_hatred': {
                'terms': [
                    'foreign plot', 'western conspiracy', 'neocolonial agenda', 'illuminati cameroon',
                    'freemason government', 'satanic leaders', 'occult practices', 'ritual sacrifices',
                    'blood money politics', 'witchcraft elections', 'traditional curses', 'ancestral punishment',
                    'foreign interference', 'cia operations', 'french manipulation', 'chinese takeover',
                    'american interests', 'british colonialism', 'german exploitation', 'israeli control'
                ],
                'severity': 'low',
                'category': 'conspiracy_hatred'
            }
        }

        # Create flat keyword list for fast lookup
        self.flat_keywords = {}
        for category, data in self.keywords.items():
            for term in data['terms']:
                self.flat_keywords[term.lower()] = {
                    'category': category,
                    'severity': data['severity']
                }

        logger.info(f"Loaded {len(self.flat_keywords)} hate speech keywords across {len(self.keywords)} categories")

    def detect_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect hate speech keywords in text

        Args:
            text: Text to analyze

        Returns:
            List of detected keywords with metadata
        """
        text_lower = text.lower()
        detections = []

        for keyword, data in self.flat_keywords.items():
            if keyword in text_lower:
                detections.append({
                    'keyword': keyword,
                    'category': data['category'],
                    'severity': data['severity']
                })

        # Sort by severity (high first)
        severity_order = {'high': 3, 'medium': 2, 'low': 1}
        detections.sort(key=lambda x: severity_order.get(x['severity'], 0), reverse=True)

        return detections

class HateSpeechAIClassifier:
    """AI-based hate speech classifier using pre-trained transformer models optimized for keyword-triggered analysis"""

    def __init__(self, model_name: str = "martin-ha/toxic-comment-model"):
        """
        Initialize the AI classifier with optimized pre-trained model

        Args:
            model_name: Hugging Face model name for hate speech detection
                       Default: martin-ha/toxic-comment-model (good for general toxicity)
                       Alternative: unitary/toxic-bert (more conservative)
        """
        try:
            # Try primary model optimized for toxicity detection
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1,
                top_k=None  # Get probabilities for all classes (replaces return_all_scores=True)
            )
            self.model_name = model_name
            logger.info(f"Loaded model: {model_name}")

        except Exception as e:
            logger.warning(f"Failed to load {model_name}, trying backup model: {e}")
            # Fallback to more widely available model
            try:
                self.classifier = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-hate-latest",
                    device=0 if torch.cuda.is_available() else -1,
                    top_k=None  # Get probabilities for all classes
                )
                self.model_name = "cardiffnlp/twitter-roberta-base-hate-latest"
                logger.info("Loaded backup model: cardiffnlp/twitter-roberta-base-hate-latest")

            except Exception as e2:
                logger.warning(f"Failed to load backup model, using basic sentiment: {e2}")
                # Final fallback
                self.classifier = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def classify_text(self, text: str) -> Tuple[bool, float, str]:
        """
        Classify text for hate speech with enhanced logic for keyword-triggered analysis

        Args:
            text: Text to classify

        Returns:
            Tuple of (is_hate_speech, confidence, explanation)
        """
        try:
            # Truncate very long texts to avoid model limits
            if len(text) > 500:
                text = text[:500] + "..."

            result = self.classifier(text)

            # Handle different model output formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    # Model returns all scores
                    result = result[0]
                else:
                    # Model returns single prediction
                    result = result[0]

            # Determine hate speech based on model type and labels
            is_hate, confidence, explanation = self._interpret_model_output(result, text)

            return is_hate, confidence, explanation

        except Exception as e:
            logger.error(f"AI classification failed for text '{text[:50]}...': {e}")
            return False, 0.0, f"Classification error: {str(e)}"

    def _interpret_model_output(self, result, original_text: str) -> Tuple[bool, float, str]:
        """
        Interpret model output based on model type and optimize for keyword-triggered context
        """

        if isinstance(result, list):
            # Multiple scores returned - find the highest confidence hate/toxic label
            hate_indicators = ['toxic', 'hate', 'offensive', 'hateful', 'harassment']
            normal_indicators = ['not_hate', 'not_toxic', 'normal', 'neutral', 'positive']

            max_hate_score = 0.0
            max_normal_score = 0.0
            hate_label = ""
            normal_label = ""

            for item in result:
                label = item['label'].lower()
                score = item['score']

                if any(indicator in label for indicator in hate_indicators):
                    if score > max_hate_score:
                        max_hate_score = score
                        hate_label = item['label']
                elif any(indicator in label for indicator in normal_indicators):
                    if score > max_normal_score:
                        max_normal_score = score
                        normal_label = item['label']

            # Decision logic: since this is keyword-triggered, be more sensitive
            if max_hate_score > 0.3:  # Lower threshold since keywords already triggered
                is_hate = True
                confidence = max_hate_score
                explanation = f"AI detected '{hate_label}' with {confidence:.2%} confidence (keyword-triggered analysis)"
            else:
                is_hate = False
                confidence = max_normal_score if max_normal_score > 0 else (1.0 - max_hate_score)
                explanation = f"AI classified as normal despite keywords with {confidence:.2%} confidence"

        else:
            # Single prediction
            label = result['label'].lower()
            score = result['score']

            # Determine if it's hate speech based on label
            hate_indicators = ['toxic', 'hate', 'offensive', 'negative', 'harassment']
            is_hate = any(indicator in label for indicator in hate_indicators)

            if is_hate:
                confidence = score
                explanation = f"AI classified as '{result['label']}' with {score:.2%} confidence"
            else:
                confidence = score
                explanation = f"AI classified as '{result['label']}' with {score:.2%} confidence"

        return is_hate, confidence, explanation

    def batch_classify(self, texts: List[str]) -> List[Tuple[bool, float, str]]:
        """
        Classify multiple texts efficiently
        """
        results = []
        for text in texts:
            results.append(self.classify_text(text))
        return results

class CameroonHateSpeechDetector:
    """Main hate speech detection system with keyword-triggered AI analysis"""

    def __init__(self):
        """Initialize the detection system"""
        logger.info("Initializing Cameroon Hate Speech Detection System...")

        # Initialize components
        self.keywords_detector = CameroonKeywordsDetector()
        self.ai_classifier = HateSpeechAIClassifier()

        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'hate_speech_detected': 0,
            'clean_content': 0,
            'keyword_triggered': 0,
            'ai_only_detected': 0,
            'category_ethnic_hatred': 0,
            'category_linguistic_discrimination': 0,
            'category_religious_hatred': 0,
            'category_political_dissent': 0,
            'category_violence_incitement': 0,
            'category_regional_discrimination': 0,
            'category_gender_discrimination': 0,
            'category_election_hatred': 0,
            'category_pidgin_hate': 0,
            'category_institutional_hatred': 0,
            'category_conspiracy_hatred': 0,
            'category_ai_detected_toxicity': 0  # Fix for AI-only detections
        }

        logger.info("✅ Detection system initialized successfully")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis
        """
        if not text or not isinstance(text, str):
            return ""

        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s\-\'\"\.!?\,\;:]', '', text)  # Remove special chars but keep punctuation

        return text

    def calculate_severity(self, keyword_detections: List[Dict], ai_confidence: float) -> str:
        """Calculate severity based on keywords and AI confidence"""
        if not keyword_detections:
            if ai_confidence > 0.8:
                return 'medium'
            else:
                return 'low'

        # Check for high severity keywords
        high_severity_count = sum(1 for d in keyword_detections if d['severity'] == 'high')
        medium_severity_count = sum(1 for d in keyword_detections if d['severity'] == 'medium')

        if high_severity_count >= 2 or (high_severity_count >= 1 and ai_confidence > 0.8):
            return 'high'
        elif high_severity_count >= 1 or medium_severity_count >= 2:
            return 'medium'
        else:
            return 'low'

    def detect_hate_speech(self, text: str) -> HateSpeechResult:
        """
        Main detection function using keyword-triggered AI analysis

        Process:
        1. Check for keywords first (fast screening)
        2. If keywords found, use AI to analyze context
        3. Combine results for final decision

        Args:
            text: Social media post text to analyze

        Returns:
            HateSpeechResult with detection details
        """
        self.stats['total_processed'] += 1

        # Preprocess text
        cleaned_text = self.preprocess_text(text)

        # Step 1: Keyword detection (fast screening)
        keyword_detections = self.keywords_detector.detect_keywords(cleaned_text)
        detected_keywords = [d['keyword'] for d in keyword_detections]

        # Step 2: AI analysis - ONLY if keywords detected or text is suspicious
        ai_is_hate, ai_confidence, ai_explanation = False, 0.0, "No AI analysis - no keywords detected"

        if keyword_detections:
            # Keywords found - use AI to analyze context and confirm
            ai_is_hate, ai_confidence, ai_explanation = self.ai_classifier.classify_text(cleaned_text)

            # If AI contradicts keywords (says it's not hate), reduce confidence
            if not ai_is_hate and ai_confidence > 0.7:
                # AI strongly disagrees with keywords - possible false positive
                final_confidence = 0.6  # Medium confidence
                is_hate_speech = True   # Trust keywords but with lower confidence
                explanation = f"Keywords detected but AI disagrees: {ai_explanation}"
            else:
                # AI agrees or is uncertain - trust the combination
                final_confidence = max(0.8, ai_confidence)
                is_hate_speech = True
                explanation = f"Keywords + AI confirmation: {ai_explanation}"

        elif len(cleaned_text.split()) > 4:  # Only analyze substantial text without keywords
            # No keywords but substantial text - quick AI check for missed hate speech
            ai_is_hate, ai_confidence, ai_explanation = self.ai_classifier.classify_text(cleaned_text)

            if ai_is_hate and ai_confidence > 0.8:  # High confidence AI detection
                is_hate_speech = True
                final_confidence = ai_confidence
                explanation = f"AI detected hate speech without keywords: {ai_explanation}"
                detected_keywords = []  # No keywords but AI found it
            else:
                # No keywords and AI doesn't think it's hate speech
                is_hate_speech = False
                final_confidence = 1.0 - ai_confidence if ai_confidence > 0 else 0.9
                explanation = f"No keywords detected, AI confirms clean: {ai_explanation}"
        else:
            # Very short text, no keywords - assume clean
            is_hate_speech = False
            final_confidence = 0.9
            explanation = "No keywords detected, text too short for AI analysis"

        # Step 3: Determine category and severity
        if keyword_detections:
            primary_category = keyword_detections[0]['category']
            severity = self.calculate_severity(keyword_detections, ai_confidence)
        elif is_hate_speech:
            primary_category = 'ai_detected_toxicity'
            severity = 'medium' if ai_confidence > 0.9 else 'low'
        else:
            primary_category = 'none'
            severity = 'none'

        # Update statistics
        if is_hate_speech:
            self.stats['hate_speech_detected'] += 1
            # Safely update category statistics
            category_key = f'category_{primary_category}'
            if category_key in self.stats:
                self.stats[category_key] += 1
            else:
                # Handle unknown categories gracefully
                logger.warning(f"Unknown category: {primary_category}, adding to ai_detected_toxicity")
                self.stats['category_ai_detected_toxicity'] += 1

            if keyword_detections:
                self.stats['keyword_triggered'] += 1
            else:
                self.stats['ai_only_detected'] += 1
        else:
            self.stats['clean_content'] += 1

        return HateSpeechResult(
            text=text,
            is_hate_speech=is_hate_speech,
            confidence=final_confidence,
            detected_keywords=detected_keywords,
            category=primary_category,
            severity=severity,
            timestamp=datetime.now(),
            explanation=explanation
        )

    def batch_detect(self, texts: List[str]) -> List[HateSpeechResult]:
        """
        Detect hate speech in multiple texts

        Args:
            texts: List of texts to analyze

        Returns:
            List of HateSpeechResult objects
        """
        results = []
        for text in texts:
            result = self.detect_hate_speech(text)
            results.append(result)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return self.stats.copy()

    def reset_statistics(self):
        """Reset statistics counters"""
        for key in self.stats:
            self.stats[key] = 0

class SocialMediaProcessor:
    """Process social media posts for hate speech detection"""

    def __init__(self, detector: CameroonHateSpeechDetector):
        self.detector = detector

    def process_post(self, post_data: Dict[str, Any]) -> Optional[HateSpeechResult]:
        """
        Process a social media post

        Args:
            post_data: Dictionary containing post information
                      Must have 'text' key, optional: 'user_id', 'platform', 'post_id'

        Returns:
            HateSpeechResult if hate speech detected, None otherwise
        """
        text = post_data.get('text', '')
        if not text:
            return None

        result = self.detector.detect_hate_speech(text)

        # Only return result if hate speech is detected
        return result if result.is_hate_speech else None

    def process_batch(self, posts: List[Dict[str, Any]]) -> List[HateSpeechResult]:
        """Process multiple posts and return only hate speech detections"""
        hate_speech_results = []

        for post in posts:
            result = self.process_post(post)
            if result:
                hate_speech_results.append(result)

        return hate_speech_results

if __name__ == "__main__":
    # Quick test of the detection system
    print("🇨🇲 Cameroon Hate Speech Detection System - Quick Test")
    print("=" * 60)

    # Initialize detector
    detector = CameroonHateSpeechDetector()

    # Test cases
    test_texts = [
        "Ces tontinards volent notre argent encore une fois",
        "Beautiful sunset in Douala today, very peaceful",
        "Paul Biya must go from power immediately",
        "We nor go gree for this rigged election",
        "Les anglofous causent toujours des problèmes ici",
        "The traditional dance was amazing at the festival"
    ]

    print("Testing keyword-triggered AI detection:\n")

    for i, text in enumerate(test_texts, 1):
        result = detector.detect_hate_speech(text)

        status = "🚨 HATE SPEECH" if result.is_hate_speech else "✅ CLEAN"
        print(f"Test {i}: {text}")
        print(f"Result: {status} (Confidence: {result.confidence:.1%})")

        if result.detected_keywords:
            print(f"Keywords: {result.detected_keywords}")

        print(f"Category: {result.category}, Severity: {result.severity}")
        print(f"Explanation: {result.explanation}")
        print("-" * 40)

    # Show statistics
    stats = detector.get_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Hate speech detected: {stats['hate_speech_detected']}")
    print(f"Clean content: {stats['clean_content']}")
    print(f"Keyword-triggered AI: {stats['keyword_triggered']}")
    print(f"AI-only detections: {stats['ai_only_detected']}")

    if stats['total_processed'] > 0:
        hate_rate = stats['hate_speech_detected'] / stats['total_processed']
        ai_usage = (stats['keyword_triggered'] + stats['ai_only_detected']) / stats['total_processed']
        print(f"Hate speech rate: {hate_rate:.1%}")
        print(f"AI usage rate: {ai_usage:.1%} (efficient!)")