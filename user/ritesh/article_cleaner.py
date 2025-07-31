import os
import glob
from time import time
from typing import Dict, Tuple, List
import pandas as pd
from multiprocessing import Pool, cpu_count
import logging
import time
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for maximum performance
class CompiledPatterns:
    def __init__(self):
        # Title filtering patterns
        self.list_title_patterns = [
            re.compile(r'^List of', re.IGNORECASE),
            re.compile(r'^Lists of', re.IGNORECASE),
            re.compile(r'filmography$', re.IGNORECASE),
            re.compile(r'discography$', re.IGNORECASE),
            re.compile(r'bibliography$', re.IGNORECASE),
            re.compile(r'\(disambiguation\)$', re.IGNORECASE),
            re.compile(r'disambiguation$', re.IGNORECASE),
            re.compile(r'^Category:', re.IGNORECASE),
            re.compile(r'^Portal:', re.IGNORECASE),
            re.compile(r'^Template:', re.IGNORECASE),
            re.compile(r'^Index of', re.IGNORECASE),
            re.compile(r'^Outline of', re.IGNORECASE),
            re.compile(r'may refer to:', re.IGNORECASE),
            re.compile(r'can refer to:', re.IGNORECASE),
            re.compile(r'index$', re.IGNORECASE),
            re.compile(r'timeline$', re.IGNORECASE),
            re.compile(r'chronology$', re.IGNORECASE)
        ]
        
        self.narrow_scope_patterns = [
            re.compile(r'at the \d{4} (Summer|Winter) Olympics', re.IGNORECASE),
            re.compile(r'at the \d{4} (World|European|Asian) (Championship|Cup)', re.IGNORECASE),
            re.compile(r'–\s*(Men\'s|Women\'s|Mixed)\s*\d+\s*(kg|m|km)', re.IGNORECASE),
            re.compile(r'–\s*(Group|Pool|Round|Heat|Semifinal|Final)\s*[A-Z]?$', re.IGNORECASE),
            re.compile(r'\d{4}\s*–\s*\d{2,4}\s*(season|series)$', re.IGNORECASE),
            re.compile(r'results$', re.IGNORECASE),
            re.compile(r'standings$', re.IGNORECASE),
            re.compile(r'qualifying$', re.IGNORECASE),
            re.compile(r'–\s*(episode|chapter|part|volume)\s*\d+', re.IGNORECASE),
            re.compile(r'–\s*(season|series)\s*\d+', re.IGNORECASE),
            re.compile(r'\d{4}\s*in\s*\w+$', re.IGNORECASE),
            re.compile(r'(nominees|winners|recipients)$', re.IGNORECASE),
            re.compile(r'(ceremony|awards)\s*\d{4}$', re.IGNORECASE),
            re.compile(r'–\s*(specification|variant|model|version)\s*[\w\d]+$', re.IGNORECASE),
            re.compile(r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d{1,2},\s+\d{4}\b', re.IGNORECASE),
            re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', re.IGNORECASE),
            re.compile(r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d{1,2}\b', re.IGNORECASE),
            re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b', re.IGNORECASE)
        ]
        
        # Content filtering patterns
        self.list_content_patterns = [
            re.compile(r'^\s*\*\s+.+$', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*#\s+.+$', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*\d+\.\s+.+$', re.IGNORECASE | re.MULTILINE)
        ]
        
        self.table_pattern = re.compile(r'\|\s*\w+\s*\|\s*\w+')
        
        self.disambiguation_indicators = [
            'may refer to:', 'can refer to:', 'disambiguation', 'for other uses',
            'for other meanings', 'not to be confused with'
        ]
        
        self.competition_indicators = [
            'elimination round', 'pool a', 'pool b', 'group stage', 'semifinals',
            'quarterfinals', 'qualifying round', 'heat 1', 'heat 2'
        ]
        
        # Explanatory content patterns (pre-compiled)
        self.explanatory_patterns = [
            re.compile(r'\b(is|are|was|were)\s+(a|an|the)?\s*\w+', re.IGNORECASE),
            re.compile(r'\b(defined\s+as|known\s+as|referred\s+to\s+as|called|termed)\b', re.IGNORECASE),
            re.compile(r'\b(means|meaning|refers\s+to|denotes|represents|indicates)\b', re.IGNORECASE),
            re.compile(r'\b(described|characterized|features|includes|contains|consists\s+of)\b', re.IGNORECASE),
            re.compile(r'\b(has|have|had|possess|possesses|exhibits|shows|displays)\b', re.IGNORECASE),
            re.compile(r'\b(appears|seems|looks|resembles|similar\s+to)\b', re.IGNORECASE),
            re.compile(r'\b(method|procedure|process|technique|algorithm|approach|system)\b', re.IGNORECASE),
            re.compile(r'\b(theory|principle|concept|law|rule|formula|equation|model)\b', re.IGNORECASE),
            re.compile(r'\b(research|study|experiment|analysis|investigation|examination|test)\b', re.IGNORECASE),
            re.compile(r'\b(developed|created|established|founded|originated|invented|designed)\b', re.IGNORECASE),
            re.compile(r'\b(discovered|identified|observed|found|noted|recorded|measured)\b', re.IGNORECASE),
            re.compile(r'\b(used|utilized|applied|employed|implemented|operated|managed)\b', re.IGNORECASE),
            re.compile(r'\b(market|industry|business|economic|financial|commercial|trade)\b', re.IGNORECASE),
            re.compile(r'\b(regulation|standard|requirement|specification|guideline|policy)\b', re.IGNORECASE),
            re.compile(r'\b(style|artistic|creative|aesthetic|cultural|traditional|genre)\b', re.IGNORECASE),
            re.compile(r'\b(interpretation|meaning|symbolism|significance|representation)\b', re.IGNORECASE),
            re.compile(r'\b(influenced|inspired|based\s+on|derived\s+from|adapted\s+from)\b', re.IGNORECASE),
            re.compile(r'\b(performance|entertainment|artistic|creative|design|visual)\b', re.IGNORECASE),
            re.compile(r'\b(gameplay|player|game|level|mechanic|strategy|puzzle|action)\b', re.IGNORECASE),
            re.compile(r'\b(multiplayer|platform|console|genre|character|story|quest)\b', re.IGNORECASE),
            re.compile(r'\b(recipe|ingredient|cooking|cuisine|flavor|preparation|dish|meal)\b', re.IGNORECASE),
            re.compile(r'\b(nutrition|nutritious|healthy|dietary|food|restaurant|chef)\b', re.IGNORECASE),
            re.compile(r'\b(treatment|therapy|diagnosis|condition|symptoms|disease|medical)\b', re.IGNORECASE),
            re.compile(r'\b(effect|affects|impact|benefit|risk|safety|danger|health)\b', re.IGNORECASE),
            re.compile(r'\b(legal|law|court|justice|crime|criminal|investigation|evidence)\b', re.IGNORECASE),
            re.compile(r'\b(penalty|prosecution|defense|judge|jury|trial|case|ruling)\b', re.IGNORECASE),
            re.compile(r'\b(vehicle|transportation|travel|route|infrastructure|traffic)\b', re.IGNORECASE),
            re.compile(r'\b(system|network|transport|journey|destination|passenger)\b', re.IGNORECASE),
            re.compile(r'\b(located|situated|positioned|found|placed|built|established)\b', re.IGNORECASE),
            re.compile(r'\b(geography|climate|environment|region|area|territory|landscape)\b', re.IGNORECASE),
            re.compile(r'\b(sport|athletic|fitness|training|exercise|competition|team)\b', re.IGNORECASE),
            re.compile(r'\b(performance|skill|technique|strategy|coach|player|athlete)\b', re.IGNORECASE),
            re.compile(r'\b(relationship|social|community|interaction|communication|behavior)\b', re.IGNORECASE),
            re.compile(r'\b(intimate|personal|private|adult|mature|emotional|psychological)\b', re.IGNORECASE),
            re.compile(r'\b(because|since|due\s+to|as\s+a\s+result|therefore|thus|hence)\b', re.IGNORECASE),
            re.compile(r'\b(during|while|when|after|before|until|following|preceding)\b', re.IGNORECASE),
            re.compile(r'\b(leads\s+to|causes|results\s+in|produces|generates|creates)\b', re.IGNORECASE),
            re.compile(r'\b(compared\s+to|unlike|similar\s+to|different\s+from|in\s+contrast)\b', re.IGNORECASE),
            re.compile(r'\b(larger|smaller|higher|lower|better|worse|more|less|greater)\b', re.IGNORECASE)
        ]
        
        # Contextual information patterns (pre-compiled)
        self.context_patterns = [
            re.compile(r'\b(background|history|origin|beginning|start|formation|foundation)\b', re.IGNORECASE),
            re.compile(r'\b(early|initial|first|original|initially|originally|prehistoric)\b', re.IGNORECASE),
            re.compile(r'\b(later|subsequent|eventually|ultimately|finally|modern|contemporary)\b', re.IGNORECASE),
            re.compile(r'\b(purpose|function|role|importance|significance|impact|influence)\b', re.IGNORECASE),
            re.compile(r'\b(designed\s+to|intended\s+to|used\s+to|serves\s+to|aims\s+to)\b', re.IGNORECASE),
            re.compile(r'\b(important|significant|notable|remarkable|famous|renowned|prominent)\b', re.IGNORECASE),
            re.compile(r'\b(methodology|technique|procedure|protocol|specification|standard)\b', re.IGNORECASE),
            re.compile(r'\b(performance|efficiency|accuracy|precision|reliability|quality)\b', re.IGNORECASE),
            re.compile(r'\b(application|implementation|deployment|usage|utilization)\b', re.IGNORECASE),
            re.compile(r'\b(market|industry|sector|economy|business|commercial|financial)\b', re.IGNORECASE),
            re.compile(r'\b(cost|price|revenue|profit|investment|budget|funding)\b', re.IGNORECASE),
            re.compile(r'\b(regulation|policy|law|legal|compliance|governance)\b', re.IGNORECASE),
            re.compile(r'\b(education|academic|research|scholarly|scientific|study|analysis)\b', re.IGNORECASE),
            re.compile(r'\b(theory|hypothesis|principle|concept|framework|model)\b', re.IGNORECASE),
            re.compile(r'\b(publication|journal|conference|peer|review|citation)\b', re.IGNORECASE),
            re.compile(r'\b(culture|cultural|society|social|community|tradition|custom)\b', re.IGNORECASE),
            re.compile(r'\b(art|artistic|literature|literary|music|musical|creative)\b', re.IGNORECASE),
            re.compile(r'\b(style|genre|movement|period|school|influence|inspiration)\b', re.IGNORECASE),
            re.compile(r'\b(entertainment|game|gaming|player|audience|viewer|fan)\b', re.IGNORECASE),
            re.compile(r'\b(story|narrative|plot|character|theme|genre|series)\b', re.IGNORECASE),
            re.compile(r'\b(platform|console|developer|publisher|release|launch)\b', re.IGNORECASE),
            re.compile(r'\b(cuisine|culinary|cooking|preparation|recipe|ingredient)\b', re.IGNORECASE),
            re.compile(r'\b(restaurant|chef|kitchen|dining|meal|dish|flavor)\b', re.IGNORECASE),
            re.compile(r'\b(nutrition|dietary|healthy|organic|traditional|regional)\b', re.IGNORECASE),
            re.compile(r'\b(health|medical|clinical|therapeutic|treatment|diagnosis)\b', re.IGNORECASE),
            re.compile(r'\b(patient|doctor|physician|hospital|healthcare|medicine)\b', re.IGNORECASE),
            re.compile(r'\b(risk|safety|side\s+effect|benefit|harm|prevention)\b', re.IGNORECASE),
            re.compile(r'\b(legal|law|court|justice|criminal|crime|investigation)\b', re.IGNORECASE),
            re.compile(r'\b(evidence|trial|case|prosecution|defense|penalty|sentence)\b', re.IGNORECASE),
            re.compile(r'\b(judge|jury|attorney|lawyer|police|enforcement)\b', re.IGNORECASE),
            re.compile(r'\b(sport|sports|athletic|fitness|training|exercise|competition)\b', re.IGNORECASE),
            re.compile(r'\b(team|player|athlete|coach|championship|tournament|league)\b', re.IGNORECASE),
            re.compile(r'\b(performance|skill|technique|strategy|physical|mental)\b', re.IGNORECASE),
            re.compile(r'\b(transportation|transport|vehicle|travel|journey|trip)\b', re.IGNORECASE),
            re.compile(r'\b(route|road|highway|infrastructure|traffic|system)\b', re.IGNORECASE),
            re.compile(r'\b(passenger|cargo|freight|logistics|network|service)\b', re.IGNORECASE),
            re.compile(r'\b(fashion|style|design|designer|clothing|apparel|wear)\b', re.IGNORECASE),
            re.compile(r'\b(beauty|cosmetic|makeup|skincare|hair|appearance)\b', re.IGNORECASE),
            re.compile(r'\b(trend|seasonal|collection|brand|luxury|affordable)\b', re.IGNORECASE),
            re.compile(r'\b(technology|software|hardware|computer|digital|electronic)\b', re.IGNORECASE),
            re.compile(r'\b(development|programming|coding|system|platform|interface)\b', re.IGNORECASE),
            re.compile(r'\b(user|developer|engineer|architect|database|network)\b', re.IGNORECASE),
            re.compile(r'\b(geographic|geographical|location|region|climate|environment)\b', re.IGNORECASE),
            re.compile(r'\b(country|city|state|province|territory|area|zone)\b', re.IGNORECASE),
            re.compile(r'\b(population|demographic|inhabitant|resident|citizen)\b', re.IGNORECASE),
            re.compile(r'\b(social|relationship|community|family|personal|intimate)\b', re.IGNORECASE),
            re.compile(r'\b(interaction|communication|behavior|psychology|emotional)\b', re.IGNORECASE),
            re.compile(r'\b(adult|mature|private|personal|individual|human)\b', re.IGNORECASE),
            re.compile(r'\b(home|house|household|domestic|interior|exterior)\b', re.IGNORECASE),
            re.compile(r'\b(hobby|craft|diy|project|collection|activity|leisure)\b', re.IGNORECASE),
            re.compile(r'\b(garden|decoration|furniture|appliance|tool|equipment)\b', re.IGNORECASE),
            re.compile(r'\b(industrial|manufacturing|production|factory|facility)\b', re.IGNORECASE),
            re.compile(r'\b(machinery|equipment|process|operation|maintenance)\b', re.IGNORECASE),
            re.compile(r'\b(worker|employee|safety|efficiency|automation)\b', re.IGNORECASE),
            re.compile(r'\b(related\s+to|connected\s+to|associated\s+with|part\s+of|member\s+of)\b', re.IGNORECASE),
            re.compile(r'\b(influenced|affected|inspired|based\s+on|derived\s+from|adapted)\b', re.IGNORECASE),
            re.compile(r'\b(leading\s+to|resulting\s+in|contributing\s+to|causing|producing)\b', re.IGNORECASE),
            re.compile(r'\b(specifically|particularly|especially|notably|mainly|primarily)\b', re.IGNORECASE),
            re.compile(r'\b(example|instance|case|illustration|demonstration|sample)\b', re.IGNORECASE),
            re.compile(r'\b(details|information|evidence|data|facts|statistics|figures)\b', re.IGNORECASE),
            re.compile(r'\b(century|decade|year|period|era|age|epoch|time|timeline)\b', re.IGNORECASE),
            re.compile(r'\b(historical|historic|ancient|medieval|renaissance|industrial)\b', re.IGNORECASE),
            re.compile(r'\b(current|modern|contemporary|recent|latest|new|emerging)\b', re.IGNORECASE)
        ]
        
        # Content depth patterns
        self.structural_patterns = [
            re.compile(r'\b\d{1,2}[,\s]+\d{4}\b'),
            re.compile(r'\b\d+\s*(kg|m|km|cm|mm|lb|ft|inches?)\b'),
            re.compile(r'\b\d+[–-]\d+\b'),
            re.compile(r'^\s*\w+:\s*\w+\s*$', re.MULTILINE),
            re.compile(r'\b(first|second|third|1st|2nd|3rd|won|lost|defeated)\b', re.IGNORECASE)
        ]
        
        self.descriptive_patterns = [
            re.compile(r'\b(described\s+as|known\s+for|characterized\s+by|famous\s+for)\b', re.IGNORECASE),
            re.compile(r'\b(because|since|due\s+to|as\s+a\s+result|therefore|however|although)\b', re.IGNORECASE),
            re.compile(r'\b(style|approach|method|technique|process|system|way)\b', re.IGNORECASE),
            re.compile(r'\b(influenced|inspired|affected|impact|effect|significance)\b', re.IGNORECASE),
            re.compile(r'\b(developed|created|designed|built|established|formed)\b', re.IGNORECASE)
        ]
        
        # Sentence splitting patterns
        self.sentence_split_pattern = re.compile(r'[.!?]+')
        self.paragraph_split_pattern = re.compile(r'\n\s*\n')
        self.word_pattern = re.compile(r'\b\w+\b')

# Initialize global patterns object
patterns = CompiledPatterns()

def build_shard_index(parquet_shards_dir) -> Dict[str, str]:
    """Build a simplified doc_id → shard_path map"""
    index = {}
    for shard_path in glob.glob(os.path.join(parquet_shards_dir, "*.parquet")):
        df = pd.read_parquet(shard_path, columns=['id'])
        for doc_id in df['id']:
            index[str(doc_id)] = shard_path
    return index

def load_topic_csv_files(topic_csv_dir: str) -> Dict[str, pd.DataFrame]:
    topic_data = {}
    for csv_file in glob.glob(os.path.join(topic_csv_dir, "*.csv")):
        topic_name = os.path.basename(csv_file).replace('.csv', '')
        df = pd.read_csv(csv_file)
        topic_data[topic_name] = df
    return topic_data

def is_list_or_disambiguation_page(title: str, text: str) -> bool:
    """Detect and filter out list articles, disambiguation pages, and narrow-scope event articles"""
    
    title_lower = title.lower().strip()
    
    # Check standard list patterns
    for pattern in patterns.list_title_patterns:
        if pattern.search(title_lower):
            return True
    
    # Check narrow-scope patterns
    for pattern in patterns.narrow_scope_patterns:
        if pattern.search(title_lower):
            return True
    
    text_lower = text.lower().strip()
    
    # Count list-like structures
    lines = text.split('\n')
    list_lines = 0
    total_content_lines = 0
    table_lines = 0
    
    for line in lines:
        line = line.strip()
        if len(line) > 10:
            total_content_lines += 1
            
            # Check for list patterns
            for pattern in patterns.list_content_patterns:
                if pattern.search(line):
                    list_lines += 1
                    break
            
            # Check for table-like content
            if patterns.table_pattern.search(line) or line.count('|') >= 3:
                table_lines += 1
    
    # If more than 50% of content lines are list-like
    if total_content_lines > 0 and (list_lines / total_content_lines) > 0.5:
        return True

    # If more than 40% of content is table-like
    if total_content_lines > 0 and (table_lines / total_content_lines) > 0.4:
        return True
        
    # Check for disambiguation indicators
    for indicator in patterns.disambiguation_indicators:
        if indicator in text_lower:
            return True

    # Check for competition/results-specific content
    competition_matches = sum(1 for indicator in patterns.competition_indicators if indicator in text_lower)
    if competition_matches >= 2:
        return True
       
    return False

def has_explanatory_content(text: str) -> bool:
    """Check if article has explanatory/descriptive content across all domains"""
    
    explanatory_matches = 0
    text_lower = text.lower()
    
    for pattern in patterns.explanatory_patterns:
        matches = len(pattern.findall(text_lower))
        explanatory_matches += matches
    
    # Normalize by text length
    words = len(text.split())
    if words < 50:
        return False
        
    explanatory_density = (explanatory_matches / words) * 100
    return explanatory_density >= 3.0

def has_coherent_paragraphs(text: str) -> bool:
    """Check for well-structured paragraphs with substantial content"""
    
    # Clean and split into paragraphs
    cleaned_text = patterns.paragraph_split_pattern.sub('\n\n', text)
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
    
    if len(paragraphs) < 2:
        return False
    
    substantial_paragraphs = 0
    for paragraph in paragraphs:
        # Skip very short paragraphs
        if len(paragraph.split()) < 20:
            continue
            
        # Check for sentence structure
        sentences = patterns.sentence_split_pattern.split(paragraph)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(valid_sentences) >= 2:
            substantial_paragraphs += 1
    
    return substantial_paragraphs >= 2

def has_contextual_information(text: str) -> bool:
    """Check for background/contextual information across all domains"""
    
    context_matches = 0
    text_lower = text.lower()
    
    for pattern in patterns.context_patterns:
        if pattern.search(text_lower):
            context_matches += 1
    
    return context_matches >= 4

def check_unique_words(text: str, min_unique_words: int = 50) -> bool:
    """Check if article has sufficient vocabulary diversity"""
    words = patterns.word_pattern.findall(text.lower())
    unique_words = set(words)
    return len(unique_words) >= min_unique_words

def check_article_length(text: str, min_chars: int = 600) -> bool:
    """Check if article meets minimum length for substantial content"""
    return len(text.strip()) >= min_chars

def check_sentence_quality(text: str) -> bool:
    """Check for proper sentence structure and variety"""
    
    # Extract sentences
    sentences = patterns.sentence_split_pattern.split(text)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    if len(valid_sentences) < 5:
        return False
    
    # Check sentence length variety
    sentence_lengths = [len(s.split()) for s in valid_sentences]
    avg_length = sum(sentence_lengths) / len(sentence_lengths)
    
    # Good articles have average sentence length between 8-30 words
    if avg_length < 8 or avg_length > 30:
        return False
    
    # Check for sentence variety
    long_sentences = [l for l in sentence_lengths if l > 15]
    return len(long_sentences) >= len(valid_sentences) * 0.3

def check_content_depth_vs_structure(text: str) -> bool:
    """Check if article has substantial descriptive content vs just structural/factual content"""
    
    text_lower = text.lower()
    sentences = patterns.sentence_split_pattern.split(text)
    
    structural_sentences = 0
    descriptive_sentences = 0
    
    for sentence in sentences:
        if len(sentence.strip()) < 15:
            continue
        
        sentence_lower = sentence.lower()
        
        # Check if sentence is primarily structural/factual
        structural_matches = sum(1 for pattern in patterns.structural_patterns 
                               if pattern.search(sentence_lower))
        
        # Check if sentence is descriptive/explanatory
        descriptive_matches = sum(1 for pattern in patterns.descriptive_patterns 
                                if pattern.search(sentence_lower))
        
        if structural_matches > 0 and descriptive_matches == 0:
            structural_sentences += 1
        elif descriptive_matches > 0:
            descriptive_sentences += 1
    
    total_sentences = structural_sentences + descriptive_sentences
    
    if total_sentences == 0:
        return False
    
    # Article should have at least 40% descriptive content
    descriptive_ratio = descriptive_sentences / total_sentences
    return descriptive_ratio >= 0.4

def check_content_quality(title: str, text: str) -> Dict[str, bool]:
    """Domain-agnostic article quality check for TOT-relevant content"""
    
    checks = {
        'is_not_list_page': not is_list_or_disambiguation_page(title, text),
        'has_min_unique_words': check_unique_words(text, min_unique_words=50),
        'has_min_length': check_article_length(text, min_chars=600),
        'has_explanatory_content': has_explanatory_content(text),
        'has_coherent_paragraphs': has_coherent_paragraphs(text),
        'has_contextual_information': has_contextual_information(text),
        'has_good_sentences': check_sentence_quality(text),
        'has_sufficient_content_depth': check_content_depth_vs_structure(text)
    }
    
    # Essential requirements
    essential_checks = [
        checks['is_not_list_page'], 
        checks['has_min_unique_words'], 
        checks['has_min_length']
    ]
    
    # Content quality checks
    content_checks = [
        checks['has_explanatory_content'],
        checks['has_coherent_paragraphs'], 
        checks['has_contextual_information'],
        checks['has_good_sentences'],
        checks['has_sufficient_content_depth']
    ]
    
    # Pass if: all essential requirements + at least 3/5 content quality checks
    checks['overall_quality'] = (
        all(essential_checks) and 
        sum(content_checks) >= 3
    )
    
    return checks

def process_articles_batch(article_batch: List[Tuple[str, str, str, str, float]]) -> Tuple[List[Dict], Dict]:
    """Process a batch of articles with comprehensive quality checks"""
    cleaned_articles = []
    stats = {
        'total_processed': 0,
        'low_quality_filtered': 0,
        'list_pages_filtered': 0,
        'quality_checks': {
            'is_not_list_page': 0, 'has_min_unique_words': 0, 'has_min_length': 0,
            'has_explanatory_content': 0, 'has_coherent_paragraphs': 0,
            'has_contextual_information': 0, 'has_good_sentences': 0,
            'has_sufficient_content_depth': 0
        }
    }
    
    for doc_id, title, url, text, confidence in article_batch:
        stats['total_processed'] += 1
        
        quality_checks = check_content_quality(title, text)
        
        # Update statistics
        for check_name, result in quality_checks.items():
            if check_name != 'overall_quality' and result:
                stats['quality_checks'][check_name] += 1
        
        if not quality_checks['is_not_list_page']:
            stats['list_pages_filtered'] += 1
        
        if quality_checks['overall_quality']:
            cleaned_articles.append({
                'id': doc_id,
                'title': title,
                'url': url,
                'text': text,
                'confidence': confidence
            })
        else:
            stats['low_quality_filtered'] += 1
    
    return cleaned_articles, stats

def process_topic_optimized(task: Tuple[str, pd.DataFrame, str, int]) -> Tuple[str, int, int, Dict]:
    """Optimized topic processing with comprehensive quality checks"""
    topic_name, df, output_dir, start_idx = task
    
    logger.info(f"Processing topic: {topic_name} with {len(df)} articles, starting from idx={start_idx}")
    
    # Sort for consistent processing
    df = df.sort_values(by='title').reset_index(drop=True)
    
    all_cleaned_articles = []
    cumulative_stats = {
        'total_processed': 0, 'low_quality_filtered': 0, 'list_pages_filtered': 0,
        'quality_checks': {
            'is_not_list_page': 0, 'has_min_unique_words': 0, 'has_min_length': 0,
            'has_explanatory_content': 0, 'has_coherent_paragraphs': 0,
            'has_contextual_information': 0, 'has_good_sentences': 0,
            'has_sufficient_content_depth': 0
        }
    }
    
    batch_size = 1000
    current_batch = []
    
    # Group by shard to minimize I/O
    shard_groups = {}
    for idx, row in df.iterrows():
        if idx < start_idx:
            continue
            
        doc_id = str(row['id'])
        if doc_id in doc_index:
            shard_path = doc_index[doc_id]
            if shard_path not in shard_groups:
                shard_groups[shard_path] = []
            shard_groups[shard_path].append((idx, doc_id, row['confidence']))
    
    # Process each shard
    for shard_path, doc_list in shard_groups.items():
        try:
            # Load entire shard once
            shard_df = pd.read_parquet(shard_path, columns=['id', 'title', 'url', 'text'])
            shard_dict = shard_df.set_index('id').to_dict('index')
            
            for idx, doc_id, confidence in doc_list:
                if doc_id in shard_dict:
                    article_data = shard_dict[doc_id]
                    current_batch.append((
                        doc_id,
                        article_data['title'],
                        article_data['url'],
                        article_data['text'],
                        confidence
                    ))
                
                # Process batch when full
                if len(current_batch) >= batch_size:
                    batch_results, batch_stats = process_articles_batch(current_batch)
                    all_cleaned_articles.extend(batch_results)
                    
                    # Update cumulative stats
                    cumulative_stats['total_processed'] += batch_stats['total_processed']
                    cumulative_stats['low_quality_filtered'] += batch_stats['low_quality_filtered']
                    cumulative_stats['list_pages_filtered'] += batch_stats['list_pages_filtered']
                    for check_name, count in batch_stats['quality_checks'].items():
                        cumulative_stats['quality_checks'][check_name] += count
                    
                    current_batch = []
                    logger.info(f"Processed {cumulative_stats['total_processed']} articles for {topic_name}, "
                              f"kept {len(all_cleaned_articles)}")
        
        except Exception as e:
            logger.error(f"Error processing shard {shard_path}: {e}")
            continue
    
    # Process remaining batch
    if current_batch:
        batch_results, batch_stats = process_articles_batch(current_batch)
        all_cleaned_articles.extend(batch_results)
        
        cumulative_stats['total_processed'] += batch_stats['total_processed']
        cumulative_stats['low_quality_filtered'] += batch_stats['low_quality_filtered']
        cumulative_stats['list_pages_filtered'] += batch_stats['list_pages_filtered']
        for check_name, count in batch_stats['quality_checks'].items():
            cumulative_stats['quality_checks'][check_name] += count
    
    # Save results in chunks
    if all_cleaned_articles:
        chunk_size = 100000
        total_chunks = (len(all_cleaned_articles) + chunk_size - 1) // chunk_size  # Ceiling division
        
        for i in range(0, len(all_cleaned_articles), chunk_size):
            chunk = all_cleaned_articles[i:i+chunk_size]
            chunk_num = i // chunk_size
            chunk_file = os.path.join(output_dir, f"{topic_name}_cleaned_chunk_{chunk_num}.parquet")
            pd.DataFrame(chunk).to_parquet(chunk_file, index=False)
            
            logger.info(f"Saved chunk {chunk_num + 1}/{total_chunks} for {topic_name}: {len(chunk)} articles")
    
    # Log detailed statistics
    logger.info(f"Completed {topic_name}:")
    logger.info(f"  Total processed: {cumulative_stats['total_processed']}")
    logger.info(f"  List/disambiguation pages filtered: {cumulative_stats['list_pages_filtered']}")
    logger.info(f"  Other low quality filtered: {cumulative_stats['low_quality_filtered'] - cumulative_stats['list_pages_filtered']}")
    logger.info(f"  Kept: {len(all_cleaned_articles)}")
    logger.info(f"  Individual check pass rates:")
    for check_name, count in cumulative_stats['quality_checks'].items():
        rate = (count / cumulative_stats['total_processed']) * 100 if cumulative_stats['total_processed'] > 0 else 0
        logger.info(f"    {check_name}: {count}/{cumulative_stats['total_processed']} ({rate:.1f}%)")
    
    return topic_name, len(all_cleaned_articles), cumulative_stats['total_processed'], cumulative_stats

if __name__ == "__main__":
    parquet_shards_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/split_parquet_shards/"
    topic_csv_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/topic_grouped_csv"
    output_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/cleaned_articles_parquet"

    os.makedirs(output_dir, exist_ok=True)

    global doc_index
    logger.info("Building article index...")
    doc_index = build_shard_index(parquet_shards_dir)
    logger.info(f"Built index for {len(doc_index)} articles")

    logger.info("Loading topic CSV files...")
    topic_data = load_topic_csv_files(topic_csv_dir)
    logger.info(f"Loaded {len(topic_data)} topics")

    # Prepare tasks - all topics with start_idx=0
    tasks = []
    for topic_name, df in topic_data.items():
        tasks.append((topic_name, df, output_dir, 0))
    
    logger.info(f"Starting optimized parallel processing of {len(tasks)} topics...")
    start_time = time.time()
    
    # # lets try on "entertainment"
    # process_topic_optimized(('entertainment', topic_data['entertainment'], output_dir, 0))
    

    # Use all available cores
    with Pool(cpu_count()) as pool:
        results = pool.map(process_topic_optimized, tasks)
    
    end_time = time.time()
    
    # Summary statistics
    total_kept = sum(result[1] for result in results)
    total_processed = sum(result[2] for result in results)
    overall_filter_rate = ((total_processed - total_kept) / total_processed * 100) if total_processed > 0 else 0
    
    logger.info(f"COMPREHENSIVE CLEANING COMPLETE in {end_time - start_time:.2f} seconds")
    logger.info(f"Total articles processed: {total_processed:,}")
    logger.info(f"Total articles kept: {total_kept:,}")
    logger.info(f"Overall filter rate: {overall_filter_rate:.1f}%")
    logger.info(f"Processing rate: {total_processed / (end_time - start_time):.0f} articles/second")
    
    # Per-topic summary
    for topic_name, kept, processed, _ in results:
        rate = (kept / processed * 100) if processed > 0 else 0
        logger.info(f"  {topic_name}: {kept:,}/{processed:,} ({rate:.1f}% kept)")