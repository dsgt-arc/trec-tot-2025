import os
import glob
from time import time
from typing import Dict, Tuple, List
import pandas as pd
from multiprocessing import Pool, cpu_count
import logging
import time
import re
import json

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
        
        # Pre-compiled disambiguation and competition patterns for faster string searches
        self.disambiguation_patterns = [
            re.compile(r'may refer to:', re.IGNORECASE),
            re.compile(r'can refer to:', re.IGNORECASE),
            re.compile(r'disambiguation', re.IGNORECASE),
            re.compile(r'for other uses', re.IGNORECASE),
            re.compile(r'for other meanings', re.IGNORECASE),
            re.compile(r'not to be confused with', re.IGNORECASE)
        ]
        
        self.competition_patterns = [
            re.compile(r'elimination round', re.IGNORECASE),
            re.compile(r'pool [ab]', re.IGNORECASE),
            re.compile(r'group stage', re.IGNORECASE),
            re.compile(r'semifinals', re.IGNORECASE),
            re.compile(r'quarterfinals', re.IGNORECASE),
            re.compile(r'qualifying round', re.IGNORECASE),
            re.compile(r'heat [12]', re.IGNORECASE)
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
        
        # Key identifying information patterns for short valuable articles (now pre-compiled)
        self.identifying_patterns = [
            # Dates and years
            re.compile(r'\b(19\d{2}|20\d{2})\b'),
            re.compile(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}', re.IGNORECASE),
            
            # Chart/ranking information
            re.compile(r'\b(no\.|number|#)\s*\d+', re.IGNORECASE),
            re.compile(r'\b(chart|billboard|hit|peaked|reached)\b', re.IGNORECASE),
            re.compile(r'\b(top|first|second|third|\d+th)\b', re.IGNORECASE),
            
            # Media/publication info
            re.compile(r'\b(album|single|ep|lp|cd|dvd|film|movie|book|novel)\b', re.IGNORECASE),
            re.compile(r'\b(released|published|aired|broadcast|premiered)\b', re.IGNORECASE),
            re.compile(r'\b(featured|appeared|included)\b', re.IGNORECASE),
            
            # People and attribution
            re.compile(r'\b(written|composed|performed|directed|produced|created|sung)\s+by\b', re.IGNORECASE),
            re.compile(r'\b(starring|featuring|with|by)\b', re.IGNORECASE),
            
            # Awards and recognition
            re.compile(r'\b(award|winner|nominated|prize|honor|recognition)\b', re.IGNORECASE),
            re.compile(r'\b(gold|platinum|certified|selling)\b', re.IGNORECASE),
            
            # Geographic and institutional info
            re.compile(r'\b(in|from|of)\s+(canada|uk|us|usa|britain|america|australia)\b', re.IGNORECASE),
            re.compile(r'\b(university|college|school|academy)\b', re.IGNORECASE)
        ]

# Initialize global patterns object
patterns = CompiledPatterns()

# Checkpoint management functions
def load_checkpoint(checkpoint_file: str) -> Dict:
    """Load checkpoint data from file"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return {}

def save_checkpoint(checkpoint_file: str, checkpoint_data: Dict):
    """Save checkpoint data to file"""
    try:
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def get_existing_progress(topic_name: str, output_dir: str) -> Tuple[int, List[str]]:
    """Get existing progress by counting articles in existing chunks"""
    chunk_pattern = os.path.join(output_dir, f"{topic_name}_cleaned_chunk_*.parquet")
    chunk_files = glob.glob(chunk_pattern)
    
    total_articles = 0
    processed_ids = set()
    
    for chunk_file in chunk_files:
        try:
            df = pd.read_parquet(chunk_file, columns=['id'])
            chunk_articles = len(df)
            total_articles += chunk_articles
            processed_ids.update(df['id'].astype(str).tolist())
            logger.info(f"Found existing chunk: {os.path.basename(chunk_file)} with {chunk_articles} articles")
        except Exception as e:
            logger.warning(f"Error reading {chunk_file}: {e}")
    
    return total_articles, list(processed_ids)

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
        
    # Check for disambiguation indicators using pre-compiled patterns
    for pattern in patterns.disambiguation_patterns:
        if pattern.search(text_lower):
            return True

    # Check for competition/results-specific content using pre-compiled patterns
    competition_matches = sum(1 for pattern in patterns.competition_patterns if pattern.search(text_lower))
    if competition_matches >= 2:
        return True
       
    return False

def has_explanatory_content(text: str) -> bool:
    """Check if article has explanatory/descriptive content across all domains (relaxed for short articles)"""
    
    explanatory_matches = 0
    text_lower = text.lower()
    
    for pattern in patterns.explanatory_patterns:
        matches = len(pattern.findall(text_lower))
        explanatory_matches += matches
    
    # Normalize by text length
    words = len(text.split())
    if words < 30:
        return False
        
    # Relaxed threshold for shorter articles
    if words < 100:
        explanatory_density = (explanatory_matches / words) * 100
        return explanatory_density >= 1.5  # Lower threshold for short articles
    else:
        explanatory_density = (explanatory_matches / words) * 100
        return explanatory_density >= 2.5  # Still lower than original 3.0

def has_coherent_paragraphs(text: str) -> bool:
    """Check for well-structured paragraphs with substantial content (relaxed for short articles)"""
    
    # Clean and split into paragraphs
    cleaned_text = patterns.paragraph_split_pattern.sub('\n\n', text)
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
    
    # For very short articles, treat the whole text as one paragraph
    if len(paragraphs) < 2:
        paragraphs = [text.strip()]
    
    substantial_paragraphs = 0
    for paragraph in paragraphs:
        # Lower word requirement for substantial paragraphs
        if len(paragraph.split()) < 15:
            continue
            
        # Check for sentence structure
        sentences = patterns.sentence_split_pattern.split(paragraph)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 8]
        
        if len(valid_sentences) >= 1:  # Just need 1 valid sentence
            substantial_paragraphs += 1
    
    return substantial_paragraphs >= 1  # At least 1 substantial paragraph

def has_contextual_information(text: str) -> bool:
    """Check for background/contextual information across all domains (relaxed for short articles)"""
    
    context_matches = 0
    text_lower = text.lower()
    
    for pattern in patterns.context_patterns:
        if pattern.search(text_lower):
            context_matches += 1
    
    # More lenient requirement based on article length
    words = len(text.split())
    if words < 100:
        return context_matches >= 2  # Just 2 contextual elements for short articles
    else:
        return context_matches >= 3  # 3 for longer articles (down from 4)

def check_unique_words(text: str, min_unique_words: int = 30) -> bool:
    """Check if article has sufficient vocabulary diversity (lowered for short valuable articles)"""
    words = patterns.word_pattern.findall(text.lower())
    unique_words = set(words)
    return len(unique_words) >= min_unique_words

def check_article_length(text: str, min_chars: int = 200) -> bool:
    """Check if article meets minimum length for substantial content (further lowered for short valuable articles)"""
    return len(text.strip()) >= min_chars

def check_sentence_quality(text: str) -> bool:
    """Check for proper sentence structure and variety (relaxed for short articles)"""
    
    # Extract sentences
    sentences = patterns.sentence_split_pattern.split(text)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    # More lenient sentence count for short articles
    words = len(text.split())
    min_sentences = 3 if words < 100 else 4  # Reduced from 5
    
    if len(valid_sentences) < min_sentences:
        return False
    
    # Check sentence length variety
    sentence_lengths = [len(s.split()) for s in valid_sentences]
    avg_length = sum(sentence_lengths) / len(sentence_lengths)
    
    # More flexible sentence length range
    if avg_length < 6 or avg_length > 35:
        return False
    
    # More lenient variety requirement for short articles
    if words < 100:
        return True  # Skip variety check for very short articles
    
    # Check for sentence variety
    long_sentences = [l for l in sentence_lengths if l > 12]  # Lowered from 15
    return len(long_sentences) >= len(valid_sentences) * 0.2  # Lowered from 0.3

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

def has_key_identifying_info(text: str) -> bool:
    """Check if short article has key identifying information typical of TOT-valuable content (optimized with pre-compiled patterns)"""
    
    text_lower = text.lower()
    
    matches = 0
    for pattern in patterns.identifying_patterns:
        if pattern.search(text_lower):
            matches += 1
    
    # Return True if we found at least 2 identifying elements
    return matches >= 2

def check_content_quality(title: str, text: str) -> Dict[str, bool]:
    """Domain-agnostic article quality check for TOT-relevant content (relaxed for short valuable articles)"""
    
    checks = {
        'is_not_list_page': not is_list_or_disambiguation_page(title, text),
        'has_min_unique_words': check_unique_words(text, min_unique_words=30),
        'has_min_length': check_article_length(text, min_chars=200),
        'has_explanatory_content': has_explanatory_content(text),
        'has_coherent_paragraphs': has_coherent_paragraphs(text),
        'has_contextual_information': has_contextual_information(text),
        'has_good_sentences': check_sentence_quality(text),
        'has_sufficient_content_depth': check_content_depth_vs_structure(text),
        'has_key_identifying_info': has_key_identifying_info(text)
    }
    
    # Essential requirements (must pass all)
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
    
    # Special case for very short articles with key identifying info
    words = len(text.split())
    if words < 80 and checks['has_key_identifying_info']:
        # Very short but information-rich articles: essentials + 1/5 content checks + identifying info
        checks['overall_quality'] = (
            all(essential_checks) and 
            sum(content_checks) >= 1
        )
    elif words < 100:
        # Short articles: essentials + 2/5 content checks
        checks['overall_quality'] = (
            all(essential_checks) and 
            sum(content_checks) >= 2
        )
    else:
        # Longer articles: essentials + 3/5 content checks  
        checks['overall_quality'] = (
            all(essential_checks) and 
            sum(content_checks) >= 3
        )
    
    return checks

def process_topic_with_checkpoints(task: Tuple[str, pd.DataFrame, str, str]) -> Tuple[str, int, int]:
    """Process a topic with checkpoint functionality"""
    topic_name, df, output_dir, checkpoint_file = task
    
    logger.info(f"Starting {topic_name} processing...")
    
    # Get existing progress
    existing_count, processed_ids = get_existing_progress(topic_name, output_dir)
    processed_ids_set = set(processed_ids)
    
    logger.info(f"{topic_name}: Found {existing_count} existing articles")
    
    # Filter out already processed articles
    df_remaining = df[~df['id'].astype(str).isin(processed_ids_set)].copy()
    
    if len(df_remaining) == 0:
        logger.info(f"{topic_name}: Already complete!")
        return topic_name, existing_count, len(df)
    
    logger.info(f"{topic_name}: Processing {len(df_remaining)} remaining articles")
    
    # Sort for consistent processing
    df_remaining = df_remaining.sort_values(by='title').reset_index(drop=True)
    
    # Processing variables
    current_chunk = []
    chunk_size = 10000
    total_processed = 0
    total_kept = existing_count
    next_chunk_num = len(glob.glob(os.path.join(output_dir, f"{topic_name}_cleaned_chunk_*.parquet")))
    
    # Group by shard for efficient I/O
    shard_groups = {}
    for idx, row in df_remaining.iterrows():
        doc_id = str(row['id'])
        if doc_id in doc_index:
            shard_path = doc_index[doc_id]
            if shard_path not in shard_groups:
                shard_groups[shard_path] = []
            shard_groups[shard_path].append((doc_id, row['confidence']))
    
    # Process each shard
    for shard_path, doc_list in shard_groups.items():
        try:
            # Load entire shard once
            shard_df = pd.read_parquet(shard_path, columns=['id', 'title', 'url', 'text'])
            shard_dict = shard_df.set_index('id').to_dict('index')
            
            for doc_id, confidence in doc_list:
                total_processed += 1
                
                if doc_id in shard_dict:
                    article_data = shard_dict[doc_id]
                    title = article_data['title']
                    text = article_data['text']
                    
                    # Quality check
                    quality_result = check_content_quality(title, text)
                    
                    if quality_result['overall_quality']:
                        current_chunk.append({
                            'id': doc_id,
                            'title': title,
                            'url': article_data['url'],
                            'text': text,
                            'confidence': confidence
                        })
                    
                    # Save chunk when full
                    if len(current_chunk) >= chunk_size:
                        chunk_file = os.path.join(output_dir, f"{topic_name}_cleaned_chunk_{next_chunk_num}.parquet")
                        pd.DataFrame(current_chunk).to_parquet(chunk_file, index=False)
                        
                        total_kept += len(current_chunk)
                        logger.info(f"{topic_name}: Saved chunk {next_chunk_num} with {len(current_chunk)} articles. "
                                  f"Total: {total_kept}, Processed: {total_processed}")
                        
                        # Update checkpoint
                        checkpoint_data = load_checkpoint(checkpoint_file)
                        checkpoint_data[topic_name] = {
                            'total_articles_kept': total_kept,
                            'total_processed': total_processed,
                            'last_chunk': next_chunk_num,
                            'timestamp': time.time(),
                            'status': 'in_progress'
                        }
                        save_checkpoint(checkpoint_file, checkpoint_data)
                        
                        current_chunk = []
                        next_chunk_num += 1
                
                # Progress logging
                if total_processed % 5000 == 0:
                    logger.info(f"{topic_name}: Processed {total_processed}, kept {total_kept - existing_count}")
        
        except Exception as e:
            logger.error(f"Error processing shard {shard_path} for {topic_name}: {e}")
            continue
    
    # Save final chunk
    if current_chunk:
        chunk_file = os.path.join(output_dir, f"{topic_name}_cleaned_chunk_{next_chunk_num}.parquet")
        pd.DataFrame(current_chunk).to_parquet(chunk_file, index=False)
        total_kept += len(current_chunk)
        logger.info(f"{topic_name}: Saved final chunk {next_chunk_num} with {len(current_chunk)} articles")
    
    # Mark as complete
    checkpoint_data = load_checkpoint(checkpoint_file)
    checkpoint_data[topic_name] = {
        'total_articles_kept': total_kept,
        'total_processed': total_processed,
        'status': 'complete',
        'completion_time': time.time()
    }
    save_checkpoint(checkpoint_file, checkpoint_data)
    
    logger.info(f"{topic_name}: COMPLETE - {total_kept} total articles kept, {total_processed} processed this run")
    return topic_name, total_kept, total_processed

if __name__ == "__main__":
    parquet_shards_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/split_parquet_shards/"
    topic_csv_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/topic_grouped_csv"
    output_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/cleaned_articles_parquet"
    checkpoint_file = os.path.join(output_dir, "progress_checkpoint.json")

    os.makedirs(output_dir, exist_ok=True)

    # Build index and load data
    # logger.info("Building article index...")
    # global doc_index
    # doc_index = build_shard_index(parquet_shards_dir)
    # logger.info(f"Built index for {len(doc_index)} articles")

    logger.info("Loading topic CSV files...")
    topic_data = load_topic_csv_files(topic_csv_dir)
    logger.info(f"Loaded {len(topic_data)} topics")

    # # Show current progress
    # checkpoint_data = load_checkpoint(checkpoint_file)
    # print("\n" + "="*80)
    # print("CURRENT PROGRESS SUMMARY")
    # print("="*80)
    
    # completed_topics = []
    # in_progress_topics = []
    # pending_topics = []
    
    # for topic_name, df in topic_data.items():
    #     total_expected = len(df)
    #     existing_count, _ = get_existing_progress(topic_name, output_dir)
        
    #     if topic_name in checkpoint_data:
    #         status = checkpoint_data[topic_name].get('status', 'unknown')
    #         kept = checkpoint_data[topic_name].get('total_articles_kept', existing_count)
            
    #         if status == 'complete':
    #             completed_topics.append((topic_name, kept, total_expected))
    #         else:
    #             in_progress_topics.append((topic_name, kept, total_expected))
    #     elif existing_count > 0:
    #         in_progress_topics.append((topic_name, existing_count, total_expected))
    #     else:
    #         pending_topics.append((topic_name, total_expected))
    
    # print(f"\n✅ COMPLETED ({len(completed_topics)}):")
    # for topic, kept, total in completed_topics:
    #     print(f"  {topic:<25} {kept:>8,} / {total:>8,} ({kept/total*100:.1f}%)")
    
    # print(f"\n🔄 IN PROGRESS ({len(in_progress_topics)}):")
    # for topic, kept, total in in_progress_topics:
    #     print(f"  {topic:<25} {kept:>8,} / {total:>8,} ({kept/total*100:.1f}%)")
    
    # print(f"\n⏳ PENDING ({len(pending_topics)}):")
    # for topic, total in pending_topics:
    #     print(f"  {topic:<25} {0:>8,} / {total:>8,} (0.0%)")
    
    # total_expected = sum(len(df) for df in topic_data.values())
    # total_kept = sum(t[1] for t in completed_topics) + sum(t[1] for t in in_progress_topics)
    # print(f"\n📊 OVERALL: {total_kept:,} / {total_expected:,} ({total_kept/total_expected*100:.1f}%)")
    # print("="*80)
    
    # # Determine what to process
    # topics_to_process = []
    # for topic_name, df in topic_data.items():
    #     if topic_name in checkpoint_data and checkpoint_data[topic_name].get('status') == 'complete':
    #         continue  # Skip completed topics
    #     topics_to_process.append((topic_name, df, output_dir, checkpoint_file))
    
    # if not topics_to_process:
    #     logger.info("All topics complete! Nothing to process.")
    #     exit(0)
    
    # logger.info(f"\nProcessing {len(topics_to_process)} topics...")
    
    # # Process topics
    # start_time = time.time()
    
    # # For testing single topic:
    # # result = process_topic_with_checkpoints(topics_to_process[0])
    # # print(f"Test result: {result}")
    # # exit(0)
    
    # with Pool(cpu_count()) as pool:
    #     results = pool.map(process_topic_with_checkpoints, topics_to_process)
    
    # end_time = time.time()
    
    # # Final summary
    # total_kept_this_run = sum(r[1] for r in results)
    # total_processed_this_run = sum(r[2] for r in results)
    
    # logger.info(f"\n🎉 SESSION COMPLETE in {end_time - start_time:.2f} seconds")
    # logger.info(f"Articles processed this session: {total_processed_this_run:,}")
    # logger.info(f"Articles kept this session: {sum(r[1] - get_existing_progress(r[0], output_dir)[0] for r in results):,}")
    # logger.info(f"Processing rate: {total_processed_this_run / (end_time - start_time):.0f} articles/second")
    
    # # Update final checkpoint
    # checkpoint_data = load_checkpoint(checkpoint_file)
    # checkpoint_data['last_session'] = {
    #     'timestamp': time.time(),
    #     'topics_processed': len(results),
    #     'total_processed': total_processed_this_run
    # }
    # save_checkpoint(checkpoint_file, checkpoint_data)
    
    # logger.info(f"Progress saved to: {checkpoint_file}")

    # lets merge all the parquet files for each topic into a single parquet file
    for topic_name, df in topic_data.items():
        logger.info(f"Merging parquet files for {topic_name}...")
        parquet_files = glob.glob(os.path.join(output_dir, f"{topic_name}_cleaned_chunk_*.parquet"))
        if len(parquet_files) > 0:
            df = pd.concat([pd.read_parquet(f) for f in parquet_files])
            df.to_parquet(os.path.join(output_dir, f"{topic_name}_cleaned.parquet"))
            # also save just the id and title per topic in a csv file
            df[['id', 'title']].to_csv(os.path.join(output_dir, f"{topic_name}_cleaned_ids_and_titles.csv"), index=False)
            # remove each parquet file
            logger.info(f"Removing {len(parquet_files)} parquet files for {topic_name}...")
            for f in parquet_files:
                os.remove(f)
            logger.info(f"Removed {len(parquet_files)} parquet files for {topic_name}")

    logger.info("All parquet files merged and cleaned")
