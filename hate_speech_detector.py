"""
Cameroon Hate Speech Detection System - FIXED VERSION

A comprehensive hate speech detection system specifically designed for Cameroonian social media content.
Uses keyword-triggered AI analysis for optimal performance and accuracy.

FIXES APPLIED:
- Improved keyword detection with accent handling and better matching
- Fixed AI model confidence interpretation
- Updated comprehensive keyword database
- Better false positive prevention
- More conservative AI thresholds

Features:
- 200+ Cameroon-specific hate speech keywords (UPDATED)
- Multi-language support (French, English, Pidgin)
- Keyword-triggered AI analysis for efficiency
- Pre-trained transformer models with better thresholds
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
import unicodedata

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
    """FIXED: Comprehensive keyword detector for Cameroonian hate speech terms"""

    def __init__(self):
        """Initialize with extensive Cameroon-specific keyword database - UPDATED"""
        self.keywords = {
            # Francophone vs Anglophone tensions
            'linguistic_discrimination': {
                'terms': [
                    'francofou', 'francofoue', 'francofolles', 'francofools',
                    'corrupt francophones', 'anglofou', 'anglofools', 'anglofous', 'anglofoue',
                    'gauche-gauche', 'gauche gauche', 'les bamenda', 'come no go', 'come-no-go',
                    'frog speakers', 'frogs', 'la republic', 'la république', 'english dogs',
                    'french slaves', 'colonial masters', 'broken english', 'pidgin speakers',
                    'francophone dictators', 'anglophone rebels', 'separatist anglos',
                    'terrorist anglophones', 'secessionist pigs', 'amba terrorists',
                    'restoration terrorists', 'ghost town supporters', 'ghost town criminals'
                ],
                'severity': 'high',
                'category': 'linguistic_discrimination'
            },

            # Regional/Tribal slurs
            'ethnic_hatred': {
                'terms': [
                    'graffi', 'ngrafi', 'graffeurs', 'nkwa', 'blackleg', 'blacklegs',
                    'tontinards', 'sardinards', 'les sardina', 'anti-sardina', 'anti sardina',
                    '99.999 sense', 'kata', '7 kata', 'sept kata', 'nkoa', 'nkoas',
                    'bamilekes', 'betis', 'ewondos', 'beti-bulu', 'fang-beti', 'bassa-bakoko',
                    'douala-littoral', 'bafia-eton', 'hausas du nord', 'kirdi', 'kirdis',
                    'moutons du nord', 'nordistes', 'sudistes', 'anglophones ennemis',
                    'francophones colonisateurs', 'separatistes', 'federalistes terroristes',
                    'biafrais', 'nigerians illegaux', 'tchadiens envahisseurs',
                    'centrafricains refugies', 'nges man', 'nges men', 'bamendaman', 'bamenda man',
                    'kumba boys'
                ],
                'severity': 'high',
                'category': 'ethnic_hatred'
            },

            # Northern/Southern divisions
            'regional_discrimination': {
                'terms': [
                    'moutons', 'wadjo', 'wadjos', 'gadamaayo', 'kirdi', 'kirdis',
                    'kaado', 'kaados', 'wari-wari', 'wari wari', 'wouri', 'bamenda grassfield',
                    'kumba boys', 'buea students', 'douala gangsters', 'yaounde bureaucrats',
                    'yaoundé bureaucrats', 'maroua backwards', 'ngaoundere cattle',
                    'garoua corrupt', 'bamenda rebels', 'limbe fishermen', 'kribi lazy',
                    'edea drunkards', 'bafoussam merchants', 'foumban traditional',
                    'tibati primitive', 'yokadouma forest', 'bertoua diamonds',
                    'sangmelima cocoa', 'ebolowa backward', 'ambam border criminals'
                ],
                'severity': 'medium',
                'category': 'regional_discrimination'
            },

            # Political terms - UPDATED WITH NEW KEYWORDS
            'political_dissent': {
                'terms': [
                    'les kamtalibans', 'kamtalibans', 'la meute', 'brigade anti-sardinards',
                    'brigade anti-tontinards', 'bas', 'bat', 'cpdm monsters', 'cpdm criminels',
                    'paul biya dictateur', 'biya must go', 'biya must die', 'down with cpdm',
                    'corrupt cpdm', 'rdpc voleurs', 'cpdm criminals', 'regime criminelle',
                    'dictature camerounaise', 'election truquee', 'élection truquée',
                    'fraud electoral', 'rigged elections', 'fake results', 'stolen votes',
                    'ballot stuffing', 'vote buying', 'constitutional council puppets',
                    'elecam corrupt', 'observers banned', 'international conspiracy',
                    'neo-colonial plot', 'maurice kamto terrorist', 'maurice kamto criminal',
                    'maurice kamto traitor', 'joshua osih enemy', 'cabral libii puppet',
                    'akere muna sellout', 'ni john fru ndi old', 'sisiku ayuk tabe separatist',
                    'sissiku terrorist', 'ambazonia criminals', 'fake news maurice kamto',
                    'kamto lies', 'kamto western puppet', 'election must be stopped',
                    'stop the election', 'boycott election', 'cpdm must fall',
                    'regime must end', 'government must go', 'no biya again', 'we nor go gree',
                    'dictature continue', 'on a volé les voix', 'man no run', 'suffrage volé'
                ],
                'severity': 'medium',
                'category': 'political_dissent'
            },

            # Violence incitement
            'violence_incitement': {
                'terms': [
                    'kill them all', 'elimination physique', 'nettoyage ethnique',
                    'guerre civile', 'civil war necessary', 'armed revolution',
                    'violence solution', 'burn everything', 'destroy anglophone regions',
                    'crush separatists', 'military solution final', 'shoot protesters',
                    'arrest opposition', 'prison for critics', 'torture enemies',
                    'kadija must pay', 'maurice kamto prison', 'joshua osih elimination',
                    'cabral libii arrest', 'akere muna traitor', 'ni john fru ndi enemy',
                    'sisiku ayuk tabe terrorist', 'biya must die', 'kamto must be eliminated',
                    'kill all opposition', 'destroy cpdm supporters', 'burn down polling stations',
                    'attack election officials', 'sabotage elections', 'civil war if biya wins',
                    'revolution against government', 'armed resistance necessary'
                ],
                'severity': 'high',
                'category': 'violence_incitement'
            },

            # Gender/sexuality slurs
            'gender_discrimination': {
                'terms': [
                    'bayangi', 'akpara', 'ashawo', 'vendeusse de piment', 'banso', 'bali',
                    'cheap girls', 'les pédés', 'les pedes', 'les bilingues', 'les ndepsos',
                    'femmes politicians incompetent', 'women weak leaders', 'girls only marriage',
                    'feminist destroyers', 'gender equality western', 'traditional women only',
                    'modern women prostitutes', 'working women neglectful', 'single mothers sinful',
                    'divorced women failures', 'widows cursed', 'barren women punished',
                    'lgbt abomination', 'homosexual western disease', 'gay marriage illegal',
                    'lesbian criminal', 'transgender mentally ill'
                ],
                'severity': 'high',
                'category': 'gender_discrimination'
            },

            # Class-based terms
            'class_discrimination': {
                'terms': [
                    'populace', 'l\'age de kumba', 'age de kumba', 'kumba age',
                    'eboa', 'kotto bass', 'nges man', 'nges men', 'guiziga',
                    'voleurs de chevre', 'voleurs de chèvre', 'goat thief', 'goat thiefs'
                ],
                'severity': 'medium',
                'category': 'class_discrimination'
            },

            # Terrorism-related slurs
            'terrorism_slurs': {
                'terms': [
                    'midin houwoum', 'sumsa djo\'ona', 'mountain people',
                    'boko haram supporters', 'muslim terrorists', 'christian crusaders',
                    'kafirs', 'infidels camerounais', 'islamic invaders', 'jihadists nordistes'
                ],
                'severity': 'high',
                'category': 'terrorism_slurs'
            },

            # 2025 Election-specific hate speech - NEW SECTION
            'election_hatred': {
                'terms': [
                    '2025 election', 'rigged election', 'tribal conflict', 'ballot box stuffed',
                    'military interference', 'vote buying', 'hacked election', 'fake results',
                    'coup rumors', 'biya dead', 'opposition jailed', 'protest planned',
                    'electoral manipulation', 'unrest anglophone', 'ghost voters',
                    'election wahala', 'vote dem don buy', 'rigged again', 'same results always',
                    'international observers waste', 'campaign restricted', 'media controlled',
                    'médias biaisés', 'medias biaises', 'biased media', 'fake news cameroon',
                    'ballot boxes stuffed', 'results predetermined', 'constitutional council biased',
                    'elecam corrupt officials', 'diaspora votes cancelled', 'registration blocked',
                    'voter cards missing', 'polling stations closed', 'military intimidation',
                    'violence during voting', 'protesters arrested', 'opposition eliminated',
                    'fake news spreading', 'social media blocked', 'internet shutdown',
                    'communications cut', 'dem don rig am', 'vote na scam', 'na who go win',
                    'na lie', 'dem go chop vote', 'e go hot for cameroon', 'di ting no clear',
                    'dey wan steal vote', 'dem wan cheat', 'fraude électorale',
                    'inscription des électeurs', 'manipulation électorale', 'voix volées',
                    'résultats truqués', 'présidentielle 2025', 'paul biya candidat',
                    'coup d\'état', 'manifestation politique', 'opposition arrêtée',
                    'violence électorale', 'parti au pouvoir', 'parti d\'opposition',
                    'électeurs fantômes', 'ingérence militaire', 'pays en crise'
                ],
                'severity': 'medium',
                'category': 'election_hatred'
            },

            # Pidgin-specific hate - UPDATED
            'pidgin_hate': {
                'terms': [
                    'dem nor fit', 'yi nor sabi', 'wuna nor get sense', 'dem na animal',
                    'yi na dog', 'wuna na goat', 'dem na pig', 'yi nor fine',
                    'ugly face dem get', 'nonsense people', 'foolish tribe', 'stupid region',
                    'useless government', 'nonsense election', 'stupid voters', 'foolish democracy',
                    'we nor go gree', 'dem don die finish', 'e go hot', 'wahala go dey',
                    'trouble go start', 'fight go begin', 'palava go full ground', 'kasala go burst',
                    'e no go work', 'who di lie', 'dey wan steal vote', 'dem wan cheat'
                ],
                'severity': 'medium',
                'category': 'pidgin_hate'
            },

            # Sports/celebrity-based
            'celebrity_hate': {
                'terms': [
                    'hibous football club', 'fidèles de la sainte eglise de tsinga',
                    'fideles de la sainte eglise de tsinga', 'églisiens', 'eglisiens'
                ],
                'severity': 'low',
                'category': 'celebrity_hate'
            },

            # Religious hatred
            'religious_hatred': {
                'terms': [
                    'animist primitives', 'feticheurs', 'féticheurs', 'sorciers',
                    'traditionalist backwards', 'church exploiters', 'mosques terrorists',
                    'mosque terrorists', 'halal killers', 'sunday hypocrites', 'ramadan criminals'
                ],
                'severity': 'high',
                'category': 'religious_hatred'
            },

            # Institutional hatred
            'institutional_hatred': {
                'terms': [
                    'corrupt ministers', 'thief governors', 'criminal prefects', 'useless parliamentarians',
                    'fake senators', 'puppet mayors', 'bought judges', 'corrupt police',
                    'military oppressors', 'gendarmerie criminals', 'bir terrorists',
                    'administrative corrupt', 'civil servants lazy', 'public servants thieves',
                    'tax collectors corrupt', 'customs criminals', 'immigration corrupt',
                    'education minister incompetent', 'health minister killer', 'finance minister thief',
                    'defense minister oppressor', 'interior minister dictator', 'justice minister corrupt'
                ],
                'severity': 'medium',
                'category': 'institutional_hatred'
            }
        }

        # Create flat keyword list for fast lookup with better normalization
        self.flat_keywords = {}
        for category, data in self.keywords.items():
            for term in data['terms']:
                normalized_term = self._normalize_text(term)
                self.flat_keywords[normalized_term] = {
                    'original': term,
                    'category': category,
                    'severity': data['severity']
                }

        logger.info(f"Loaded {len(self.flat_keywords)} hate speech keywords across {len(self.keywords)} categories")

    def _normalize_text(self, text: str) -> str:
        """
        FIXED: Normalize text for better keyword matching
        - Remove accents and diacritics
        - Convert to lowercase
        - Handle special characters
        """
        if not text:
            return ""

        # Remove accents and diacritics
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')

        # Convert to lowercase
        text = text.lower()

        # Replace common variations
        text = text.replace('é', 'e').replace('è', 'e').replace('ê', 'e')
        text = text.replace('à', 'a').replace('â', 'a')
        text = text.replace('ô', 'o').replace('ö', 'o')
        text = text.replace('ù', 'u').replace('û', 'u')
        text = text.replace('ç', 'c')

        # Remove extra spaces and punctuation for matching
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def detect_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        FIXED: Detect hate speech keywords in text with better matching for mixed-language content
        """
        if not text:
            return []

        normalized_text = self._normalize_text(text)
        detections = []

        for keyword, data in self.flat_keywords.items():
            # Check for exact matches and word boundary matches
            if keyword in normalized_text:
                # For longer terms (3+ chars), use word boundary matching
                if len(keyword) > 3:
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, normalized_text):
                        detections.append({
                            'keyword': data['original'],
                            'category': data['category'],
                            'severity': data['severity']
                        })
                else:
                    # For short terms, use exact match
                    detections.append({
                        'keyword': data['original'],
                        'category': data['category'],
                        'severity': data['severity']
                    })

        # SPECIAL HANDLING for mixed-language media bias terms
        # Handle "Media biaisés" and similar mixed-language cases
        media_bias_patterns = [
            (r'\bmedia\s+biaise[ds]?\b', 'media biaisés', 'election_hatred', 'medium'),
            (r'\bmedias?\s+biaise[ds]?\b', 'médias biaisés', 'election_hatred', 'medium'),
            (r'\bbias(ed)?\s+media\b', 'biased media', 'election_hatred', 'medium'),
            (r'\bfake\s+news\s+cameroo[un]\b', 'fake news cameroon', 'election_hatred', 'medium')
        ]

        for pattern, keyword, category, severity in media_bias_patterns:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                detections.append({
                    'keyword': keyword,
                    'category': category,
                    'severity': severity
                })

        # Remove duplicates and sort by severity
        seen = set()
        unique_detections = []
        for detection in detections:
            key = (detection['keyword'], detection['category'])
            if key not in seen:
                seen.add(key)
                unique_detections.append(detection)

        # Sort by severity (high first)
        severity_order = {'high': 3, 'medium': 2, 'low': 1}
        unique_detections.sort(key=lambda x: severity_order.get(x['severity'], 0), reverse=True)

        return unique_detections

class HateSpeechAIClassifier:
    """FIXED: AI-based hate speech classifier with better model selection and thresholds"""

    def __init__(self, model_name: str = "martin-ha/toxic-comment-model"):
        """
        FIXED: Initialize with better model selection prioritizing effectiveness over conservatism
        """
        try:
            # Try martin-ha/toxic-comment-model first (better for general toxicity)
            self.classifier = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",
                device=0 if torch.cuda.is_available() else -1,
                top_k=None
            )
            self.model_name = "martin-ha/toxic-comment-model"
            logger.info(f"Loaded primary model: martin-ha/toxic-comment-model")

        except Exception as e:
            logger.warning(f"Failed to load martin-ha/toxic-comment-model, trying cardiff: {e}")
            try:
                # Fallback to cardiff model (good balance)
                self.classifier = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-hate-latest",
                    device=0 if torch.cuda.is_available() else -1,
                    top_k=None
                )
                self.model_name = "cardiffnlp/twitter-roberta-base-hate-latest"
                logger.info("Loaded fallback model: cardiffnlp/twitter-roberta-base-hate-latest")

            except Exception as e2:
                logger.warning(f"Failed to load hate detection model, using toxic-bert: {e2}")
                # Last resort - toxic-bert (but it's too conservative)
                self.classifier = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    device=0 if torch.cuda.is_available() else -1,
                    top_k=None
                )
                self.model_name = "unitary/toxic-bert"
                logger.warning("Using toxic-bert - may be too conservative for Cameroon context")

    def classify_text(self, text: str) -> Tuple[bool, float, str]:
        """
        FIXED: Classify text with more conservative thresholds and better interpretation
        """
        try:
            # Truncate very long texts
            if len(text) > 500:
                text = text[:500] + "..."

            result = self.classifier(text)

            # Handle different output formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    result = result[0]
                else:
                    result = result[0]

            # FIXED: More conservative interpretation
            is_hate, confidence, explanation = self._interpret_model_output_conservative(result, text)

            return is_hate, confidence, explanation

        except Exception as e:
            logger.error(f"AI classification failed for text '{text[:50]}...': {e}")
            return False, 0.0, f"Classification error: {str(e)}"

    def _interpret_model_output_conservative(self, result, original_text: str) -> Tuple[bool, float, str]:
        """
        FIXED: Better interpretation that accounts for model limitations with local context
        """

        if isinstance(result, list):
            # Multiple scores - find hate/toxic indicators
            hate_indicators = ['toxic', 'hate', 'offensive', 'hateful', 'harassment', 'threat']
            safe_indicators = ['not_hate', 'not_toxic', 'normal', 'neutral', 'safe']

            max_hate_score = 0.0
            max_safe_score = 0.0
            hate_label = ""

            for item in result:
                label = item['label'].lower()
                score = item['score']

                if any(indicator in label for indicator in hate_indicators):
                    if score > max_hate_score:
                        max_hate_score = score
                        hate_label = item['label']
                elif any(indicator in label for indicator in safe_indicators):
                    if score > max_safe_score:
                        max_safe_score = score

            # FIXED: More balanced threshold for multi-class models
            if max_hate_score > 0.6:  # Lowered from 0.8 to 0.6
                is_hate = True
                confidence = max_hate_score
                explanation = f"AI detected hate speech: '{hate_label}' ({confidence:.1%} confidence)"
            else:
                is_hate = False
                confidence = max_safe_score if max_safe_score > 0 else (1.0 - max_hate_score)
                explanation = f"AI classified as safe: confidence ({confidence:.1%})"

        else:
            # Single prediction
            label = result['label'].lower()
            score = result['score']

            hate_indicators = ['toxic', 'hate', 'offensive', 'negative', 'harassment']

            # FIXED: More balanced thresholds
            if any(indicator in label for indicator in hate_indicators):
                if score > 0.7:  # Lowered from 0.85 to 0.7
                    is_hate = True
                    confidence = score
                    explanation = f"AI detected hate speech: '{result['label']}' ({score:.1%})"
                else:
                    is_hate = False
                    confidence = 1.0 - score
                    explanation = f"AI uncertain about hate speech: '{result['label']}' ({score:.1%})"
            else:
                is_hate = False
                confidence = score
                explanation = f"AI classified as safe: '{result['label']}' ({score:.1%})"

        return is_hate, confidence, explanation

class CameroonHateSpeechDetector:
    """FIXED: Main hate speech detection system with improved logic"""

    def __init__(self):
        """Initialize the detection system"""
        logger.info("Initializing FIXED Cameroon Hate Speech Detection System...")

        # Initialize components
        self.keywords_detector = CameroonKeywordsDetector()
        self.ai_classifier = HateSpeechAIClassifier()

        # Enhanced innocent content indicators
        self.innocent_indicators = [
            'cuisine', 'food', 'recipe', 'cooking', 'delicious', 'excellent', 'good',
            'beautiful', 'ceremony', 'graduation', 'wedding', 'festival', 'celebration',
            'market', 'school', 'university', 'hospital', 'church', 'mosque',
            'football', 'sport', 'music', 'dance', 'culture', 'tradition', 'traditional',
            'weather', 'season', 'rain', 'sunshine', 'family', 'children',
            'birthday', 'holiday', 'harvest', 'peaceful', 'inspiring', 'amazing',
            'sunset', 'sunrise', 'nature', 'landscape', 'tourism', 'travel'
        ]

        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'hate_speech_detected': 0,
            'clean_content': 0,
            'keyword_triggered': 0,
            'ai_only_detected': 0,
            'false_positive_prevention': 0,
            'category_linguistic_discrimination': 0,
            'category_ethnic_hatred': 0,
            'category_regional_discrimination': 0,
            'category_political_dissent': 0,
            'category_violence_incitement': 0,
            'category_gender_discrimination': 0,
            'category_class_discrimination': 0,
            'category_terrorism_slurs': 0,
            'category_election_hatred': 0,
            'category_pidgin_hate': 0,
            'category_celebrity_hate': 0,
            'category_religious_hatred': 0,
            'category_institutional_hatred': 0,
            'category_ai_detected_toxicity': 0
        }

        logger.info("✅ FIXED Detection system initialized successfully")

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        if not text or not isinstance(text, str):
            return ""

        # Basic cleaning while preserving accents for keyword detection
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

        return text

    def is_likely_innocent(self, text: str) -> bool:
        """
        FIXED: Enhanced innocent content detection
        """
        text_lower = text.lower()

        # Check for innocent indicators
        innocent_count = sum(1 for indicator in self.innocent_indicators if indicator in text_lower)

        # If multiple innocent indicators, likely safe
        if innocent_count >= 2:
            return True

        # Check for specific safe patterns
        safe_patterns = [
            r'\b(cuisine|food|recipe|cooking).*(excellent|good|delicious|amazing)\b',
            r'\b(beautiful|peaceful|inspiring).*(day|evening|ceremony|festival)\b',
            r'\b(traditional|cultural).*(dance|music|ceremony|festival)\b',
            r'\b(school|university|hospital|church|mosque)\b',
            r'\b(weather|season|harvest|market)\b'
        ]

        for pattern in safe_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def is_cameroon_specific_hate(self, keyword_detections: List[Dict]) -> bool:
        """
        Check if the detected keywords are Cameroon-specific hate speech
        that the AI model might not understand due to lack of local context
        """
        cameroon_specific_terms = {
            'tontinards', 'sardinards', 'anglofous', 'francofou', 'graffi', 'wadjo',
            'les kamtalibans', 'kamtalibans', 'nges man', 'bamenda', 'come no go',
            'moutons', 'kirdi', 'beti', 'ewondo', 'hausa', 'bamileke', 'nkwa',
            'medias biaises', 'media biaises', 'we nor go gree', 'dem don rig',
            'vote na scam', 'e go hot', 'rigged election', 'biya must go'
        }

        detected_terms = [d['keyword'].lower() for d in keyword_detections]

        # Check if any detected keyword is Cameroon-specific
        return any(
            any(specific_term in detected_term for specific_term in cameroon_specific_terms)
            for detected_term in detected_terms
        )

    def detect_hate_speech(self, text: str) -> HateSpeechResult:
        """
        FIXED: Main detection with improved logic and better false positive prevention
        """
        self.stats['total_processed'] += 1

        # Preprocess text
        cleaned_text = self.preprocess_text(text)

        # FIXED: Enhanced innocent content check
        if self.is_likely_innocent(cleaned_text):
            self.stats['clean_content'] += 1
            self.stats['false_positive_prevention'] += 1
            return HateSpeechResult(
                text=text,
                is_hate_speech=False,
                confidence=0.95,
                detected_keywords=[],
                category='none',
                severity='none',
                timestamp=datetime.now(),
                explanation="Content identified as innocent (cultural, food, peaceful, etc.)"
            )

        # Step 1: FIXED keyword detection
        keyword_detections = self.keywords_detector.detect_keywords(cleaned_text)
        detected_keywords = [d['keyword'] for d in keyword_detections]

        # Step 2: AI analysis logic
        ai_is_hate, ai_confidence, ai_explanation = False, 0.0, "No AI analysis needed"

        if keyword_detections:
            # Keywords found - use AI to verify context but trust keywords for Cameroon-specific terms
            ai_is_hate, ai_confidence, ai_explanation = self.ai_classifier.classify_text(cleaned_text)

            # Check if we have high-severity Cameroon-specific keywords
            high_severity_keywords = [d for d in keyword_detections if d['severity'] == 'high']
            has_cameroon_specific = self.is_cameroon_specific_hate(keyword_detections)

            # DEBUG: Log decision process
            logger.debug(f"Keyword analysis: {len(keyword_detections)} keywords, {len(high_severity_keywords)} high severity")
            logger.debug(f"Cameroon-specific: {has_cameroon_specific}")
            logger.debug(f"AI result: {ai_is_hate} with {ai_confidence:.1%} confidence")

            # FIXED: Balanced decision logic that trusts keywords for Cameroon context
            if ai_is_hate and ai_confidence > 0.7:
                # AI agrees with keywords
                is_hate_speech = True
                final_confidence = ai_confidence
                explanation = f"Keywords + AI confirmation: {ai_explanation}"
            elif high_severity_keywords and has_cameroon_specific:
                # High severity Cameroon-specific terms - trust keywords over AI
                is_hate_speech = True
                final_confidence = 0.8
                explanation = f"High-severity Cameroon-specific keywords detected: {[d['keyword'] for d in high_severity_keywords]} (AI may not understand local context)"
            elif len(keyword_detections) >= 2:
                # Multiple keywords detected - likely hate speech
                is_hate_speech = True
                final_confidence = 0.75
                explanation = f"Multiple keywords detected: {[d['keyword'] for d in keyword_detections]} (AI: {ai_explanation})"
            elif not ai_is_hate and ai_confidence > 0.95 and not has_cameroon_specific:
                # AI very strongly disagrees and no Cameroon-specific terms - possible false positive
                is_hate_speech = False
                final_confidence = ai_confidence
                explanation = f"AI very strongly disagrees with keywords: {ai_explanation}"
                self.stats['false_positive_prevention'] += 1
            else:
                # Default: trust keywords with moderate confidence
                is_hate_speech = True
                final_confidence = 0.7
                explanation = f"Keywords detected: {[d['keyword'] for d in keyword_detections]} (AI uncertain about local context)"

        elif len(cleaned_text.split()) > 6:  # Only check substantial text
            # No keywords - quick AI check for missed toxicity
            ai_is_hate, ai_confidence, ai_explanation = self.ai_classifier.classify_text(cleaned_text)

            # FIXED: Very conservative threshold when no keywords
            if ai_is_hate and ai_confidence > 0.95:  # Extremely high threshold
                is_hate_speech = True
                final_confidence = ai_confidence
                explanation = f"AI detected very strong toxicity without keywords: {ai_explanation}"
            else:
                is_hate_speech = False
                final_confidence = 1.0 - ai_confidence if ai_confidence > 0 else 0.9
                explanation = f"No keywords, AI analysis inconclusive: {ai_explanation}"
        else:
            # Short text, no keywords - assume clean
            is_hate_speech = False
            final_confidence = 0.9
            explanation = "Short text, no keywords detected"

        # Step 3: Determine category and severity
        if keyword_detections:
            primary_category = keyword_detections[0]['category']
            severity = self._calculate_severity(keyword_detections, ai_confidence)
        elif is_hate_speech:
            primary_category = 'ai_detected_toxicity'
            severity = 'medium' if ai_confidence > 0.9 else 'low'
        else:
            primary_category = 'none'
            severity = 'none'

        # Update statistics
        if is_hate_speech:
            self.stats['hate_speech_detected'] += 1
            category_key = f'category_{primary_category}'
            if category_key in self.stats:
                self.stats[category_key] += 1

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

    def _calculate_severity(self, keyword_detections: List[Dict], ai_confidence: float) -> str:
        """Calculate severity based on keywords and AI confidence"""
        if not keyword_detections:
            return 'low'

        high_severity_count = sum(1 for d in keyword_detections if d['severity'] == 'high')
        medium_severity_count = sum(1 for d in keyword_detections if d['severity'] == 'medium')

        if high_severity_count >= 2 or (high_severity_count >= 1 and ai_confidence > 0.8):
            return 'high'
        elif high_severity_count >= 1 or medium_severity_count >= 2:
            return 'medium'
        else:
            return 'low'

    def batch_detect(self, texts: List[str]) -> List[HateSpeechResult]:
        """Detect hate speech in multiple texts"""
        results = []
        for text in texts:
            result = self.detect_hate_speech(text)
            results.append(result)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        stats = self.stats.copy()
        if stats['total_processed'] > 0:
            stats['hate_rate'] = stats['hate_speech_detected'] / stats['total_processed']
            stats['false_positive_prevention_rate'] = stats['false_positive_prevention'] / stats['total_processed']
        return stats

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

# TESTING CODE
if __name__ == "__main__":
    print("🇨🇲 FIXED Cameroon Hate Speech Detection System - Test")
    print("=" * 60)

    # Initialize detector
    detector = CameroonHateSpeechDetector()

    # Test cases including the problematic ones
    test_texts = [
        "Les Kamtalibans détruisent notre pays",           # Should detect - political hate
        "Media biaisés refusing to cover opposition",      # Should detect - election hate
        "La cuisine camerounaise est excellente",          # Should be CLEAN - food content
        "Ces tontinards volent notre argent encore",       # Should detect - ethnic slur
        "Beautiful sunset in Douala today, very peaceful", # Should be clean - innocent
        "Paul Biya must go from power immediately",        # Should detect - political
        "We nor go gree for this rigged election",         # Should detect - pidgin political
        "Ces anglofous causent toujours des problèmes",    # Should detect - linguistic hate
        "The traditional dance was amazing at the festival", # Should be clean - cultural
        "Good harvest expected this season",               # Should be clean - innocent
        "Maurice Kamto gave a speech yesterday",           # Should be clean - normal news
        "Vote buying reported in Northern regions",        # Should detect - election hate
        "Amazing football match yesterday in Yaoundé",     # Should be clean - sports
        "Ces wadjo volent nos ressources naturelles"       # Should detect - regional hate
    ]

    print("Testing FIXED keyword-triggered AI detection:\n")

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
    print(f"False positive prevention: {stats['false_positive_prevention']}")
    print(f"Keyword-triggered AI: {stats['keyword_triggered']}")
    print(f"AI-only detections: {stats['ai_only_detected']}")

    if stats['total_processed'] > 0:
        hate_rate = stats['hate_speech_detected'] / stats['total_processed']
        prevention_rate = stats['false_positive_prevention'] / stats['total_processed']
        print(f"Hate speech rate: {hate_rate:.1%}")
        print(f"False positive prevention rate: {prevention_rate:.1%}")