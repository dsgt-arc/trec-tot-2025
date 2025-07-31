from asyncio import tasks
import os
import glob
from time import time
from typing import Dict, Tuple, List
from numpy import save
import pandas as pd
from multiprocessing import Pool, cpu_count
import logging
import time
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Build a doc_id → (shard_path, row_index) map
def build_shard_index(parquet_shards_dir) -> Dict[str, Tuple[str, int]]:
    index = {}
    for shard_path in glob.glob(os.path.join(parquet_shards_dir, "*.parquet")):
        df = pd.read_parquet(shard_path, columns=['id'])  # just load IDs
        for i, doc_id in enumerate(df['id']):
            index[str(doc_id)] = (shard_path, i)
    return index

# Function to load topic CSV files
def load_topic_csv_files(topic_csv_dir: str) -> Dict[str, pd.DataFrame]:
    topic_data = {}
    for csv_file in glob.glob(os.path.join(topic_csv_dir, "*.csv")):
        topic_name = os.path.basename(csv_file).replace('.csv', '')
        df = pd.read_csv(csv_file)
        topic_data[topic_name] = df
    return topic_data

def is_list_or_disambiguation_page(title: str, text: str) -> bool:
    """Detect and filter out list articles and disambiguation pages - not useful for TOT"""
    
    # Check title patterns for lists and meta-pages
    list_title_patterns = [
        r'^List of',
        r'^Lists of',
        r'filmography$',
        r'discography$',
        r'bibliography$',
        r'\(disambiguation\)$',
        r'disambiguation$',
        r'^Category:',
        r'^Portal:',
        r'^Template:',
        r'^Index of',
        r'^Outline of',
        r'may refer to:',
        r'can refer to:',
        r'index$',
        r'timeline$',
        r'chronology$'
    ]

    # Add patterns for narrow-scope event/competition articles (not useful for TOT)
    narrow_scope_patterns = [
        # Specific sports events/competitions
        r'at the \d{4} (Summer|Winter) Olympics',
        r'at the \d{4} (World|European|Asian) (Championship|Cup)',
        r'–\s*(Men\'s|Women\'s|Mixed)\s*\d+\s*(kg|m|km)',  # Weight classes, distances
        r'–\s*(Group|Pool|Round|Heat|Semifinal|Final)\s*[A-Z]?$',
        r'\d{4}\s*–\s*\d{2,4}\s*(season|series)$',
        r'results$',
        r'standings$',
        r'qualifying$',
        
        # Very specific sub-articles
        r'–\s*(episode|chapter|part|volume)\s*\d+',
        r'–\s*(season|series)\s*\d+',
        r'\d{4}\s*in\s*\w+$',  # "2020 in sports", "1995 in film"
        
        # Award/competition specific categories
        r'(nominees|winners|recipients)$',
        r'(ceremony|awards)\s*\d{4}$',
        
        # Very technical sub-specifications
        r'–\s*(specification|variant|model|version)\s*[\w\d]+$',

        # Days of the week or months in titles separated by dates e.g., April 11
        r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d{1,2},\s+\d{4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
        r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d{1,2}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b',
    ]
    

    title_lower = title.lower().strip()
    for pattern in list_title_patterns:
        if re.search(pattern, title_lower, re.IGNORECASE):
            return True

    # Check narrow-scope patterns
    for pattern in narrow_scope_patterns:
        if re.search(pattern, title_lower, re.IGNORECASE):
            return True
            
    # Check content patterns for lists/disambiguation
    text_lower = text.lower().strip()

    # Count list-like structures
    list_patterns = [
        r'^\s*\*\s+.+$',  # Bullet points
        r'^\s*#\s+.+$',   # Numbered lists  
        r'^\s*\d+\.\s+.+$',  # Numbered items
    ]
    

    lines = text.split('\n')
    list_lines = 0
    total_content_lines = 0
    table_lines = 0
    
    for line in lines:
        line = line.strip()
        if len(line) > 10:  # Only count substantial lines
            total_content_lines += 1
            
            # Check for list patterns
            for pattern in list_patterns:
                if re.search(pattern, line, re.IGNORECASE | re.MULTILINE):
                    list_lines += 1
                    break
            
            # Check for table-like content (common in competition results)
            if re.search(r'\|\s*\w+\s*\|\s*\w+', line) or line.count('|') >= 3:
                table_lines += 1
    
    # If more than 50% of content lines are list-like, probably a list article
    if total_content_lines > 0 and (list_lines / total_content_lines) > 0.5:
        return True

    # If more than 40% of content is table-like (common in results pages)
    if total_content_lines > 0 and (table_lines / total_content_lines) > 0.4:
        return True
        
    # Check for disambiguation indicators in text
    disambiguation_indicators = [
        'may refer to:',
        'can refer to:',
        'disambiguation',
        'for other uses',
        'for other meanings',
        'not to be confused with'
    ]
    
    for indicator in disambiguation_indicators:
        if indicator in text_lower:
            return True

    # Check for competition/results-specific content
    competition_indicators = [
        'elimination round',
        'pool a',
        'pool b', 
        'group stage',
        'semifinals',
        'quarterfinals',
        'qualifying round',
        'heat 1',
        'heat 2'
    ]
    
    competition_matches = sum(1 for indicator in competition_indicators if indicator in text_lower)
    if competition_matches >= 2:  # Multiple competition-specific terms
        return True
       
    return False

def has_explanatory_content(text: str) -> bool:
    """Check if article has explanatory/descriptive content across all domains"""
    
    # Universal explanatory patterns
    explanatory_patterns = [
        # Definitional content (all domains)
        r'\b(is|are|was|were)\s+(a|an|the)?\s*\w+',
        r'\b(defined\s+as|known\s+as|referred\s+to\s+as|called|termed)\b',
        r'\b(means|meaning|refers\s+to|denotes|represents|indicates)\b',
        
        # Descriptive content (all domains)
        r'\b(described|characterized|features|includes|contains|consists\s+of)\b',
        r'\b(has|have|had|possess|possesses|exhibits|shows|displays)\b',
        r'\b(appears|seems|looks|resembles|similar\s+to)\b',
        
        # Technical/Scientific patterns
        r'\b(method|procedure|process|technique|algorithm|approach|system)\b',
        r'\b(theory|principle|concept|law|rule|formula|equation|model)\b',
        r'\b(research|study|experiment|analysis|investigation|examination|test)\b',
        r'\b(developed|created|established|founded|originated|invented|designed)\b',
        r'\b(discovered|identified|observed|found|noted|recorded|measured)\b',
        
        # Business/Professional patterns
        r'\b(used|utilized|applied|employed|implemented|operated|managed)\b',
        r'\b(market|industry|business|economic|financial|commercial|trade)\b',
        r'\b(regulation|standard|requirement|specification|guideline|policy)\b',
        
        # Creative/Entertainment patterns
        r'\b(style|artistic|creative|aesthetic|cultural|traditional|genre)\b',
        r'\b(interpretation|meaning|symbolism|significance|representation)\b',
        r'\b(influenced|inspired|based\s+on|derived\s+from|adapted\s+from)\b',
        r'\b(performance|entertainment|artistic|creative|design|visual)\b',
        
        # Gaming patterns
        r'\b(gameplay|player|game|level|mechanic|strategy|puzzle|action)\b',
        r'\b(multiplayer|platform|console|genre|character|story|quest)\b',
        
        # Food/Dining patterns
        r'\b(recipe|ingredient|cooking|cuisine|flavor|preparation|dish|meal)\b',
        r'\b(nutrition|nutritious|healthy|dietary|food|restaurant|chef)\b',
        
        # Health/Medical patterns
        r'\b(treatment|therapy|diagnosis|condition|symptoms|disease|medical)\b',
        r'\b(effect|affects|impact|benefit|risk|safety|danger|health)\b',
        
        # Legal/Crime patterns
        r'\b(legal|law|court|justice|crime|criminal|investigation|evidence)\b',
        r'\b(penalty|prosecution|defense|judge|jury|trial|case|ruling)\b',
        
        # Transportation patterns
        r'\b(vehicle|transportation|travel|route|infrastructure|traffic)\b',
        r'\b(system|network|transport|journey|destination|passenger)\b',
        
        # Geographic/Location patterns
        r'\b(located|situated|positioned|found|placed|built|established)\b',
        r'\b(geography|climate|environment|region|area|territory|landscape)\b',
        
        # Sports/Fitness patterns
        r'\b(sport|athletic|fitness|training|exercise|competition|team)\b',
        r'\b(performance|skill|technique|strategy|coach|player|athlete)\b',
        
        # Relationship/Social patterns (including adult content)
        r'\b(relationship|social|community|interaction|communication|behavior)\b',
        r'\b(intimate|personal|private|adult|mature|emotional|psychological)\b',
        
        # Causal and temporal relationships (all domains)
        r'\b(because|since|due\s+to|as\s+a\s+result|therefore|thus|hence)\b',
        r'\b(during|while|when|after|before|until|following|preceding)\b',
        r'\b(leads\s+to|causes|results\s+in|produces|generates|creates)\b',
        
        # Comparative content (all domains)
        r'\b(compared\s+to|unlike|similar\s+to|different\s+from|in\s+contrast)\b',
        r'\b(larger|smaller|higher|lower|better|worse|more|less|greater)\b',
    ]
    
    explanatory_matches = 0
    text_lower = text.lower()
    
    for pattern in explanatory_patterns:
        matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
        explanatory_matches += matches
    
    # Normalize by text length (matches per 100 words)
    words = len(text.split())
    if words < 50:
        return False
        
    explanatory_density = (explanatory_matches / words) * 100
    return explanatory_density >= 3.0  # At least 3% of words are explanatory

def has_coherent_paragraphs(text: str) -> bool:
    """Check for well-structured paragraphs with substantial content"""
    
    # Clean and split into paragraphs
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', text)
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
    
    if len(paragraphs) < 2:
        return False
    
    substantial_paragraphs = 0
    for paragraph in paragraphs:
        # Skip very short paragraphs
        if len(paragraph.split()) < 20:
            continue
            
        # Check for sentence structure (has proper punctuation)
        sentences = re.split(r'[.!?]+', paragraph)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(valid_sentences) >= 2:  # At least 2 proper sentences
            substantial_paragraphs += 1
    
    # At least 2 substantial paragraphs
    return substantial_paragraphs >= 2

def has_contextual_information(text: str) -> bool:
    """Check for background/contextual information across all domains"""
    
    # Domain-comprehensive context patterns
    context_patterns = [
        # Background/history (all domains)
        r'\b(background|history|origin|beginning|start|formation|foundation)\b',
        r'\b(early|initial|first|original|initially|originally|prehistoric)\b',
        r'\b(later|subsequent|eventually|ultimately|finally|modern|contemporary)\b',
        
        # Purpose/function/importance (all domains)
        r'\b(purpose|function|role|importance|significance|impact|influence)\b',
        r'\b(designed\s+to|intended\s+to|used\s+to|serves\s+to|aims\s+to)\b',
        r'\b(important|significant|notable|remarkable|famous|renowned|prominent)\b',
        
        # Technical/Scientific context
        r'\b(methodology|technique|procedure|protocol|specification|standard)\b',
        r'\b(performance|efficiency|accuracy|precision|reliability|quality)\b',
        r'\b(application|implementation|deployment|usage|utilization)\b',
        
        # Business/Economic context
        r'\b(market|industry|sector|economy|business|commercial|financial)\b',
        r'\b(cost|price|revenue|profit|investment|budget|funding)\b',
        r'\b(regulation|policy|law|legal|compliance|governance)\b',
        
        # Educational/Academic context
        r'\b(education|academic|research|scholarly|scientific|study|analysis)\b',
        r'\b(theory|hypothesis|principle|concept|framework|model)\b',
        r'\b(publication|journal|conference|peer|review|citation)\b',
        
        # Cultural/Creative context
        r'\b(culture|cultural|society|social|community|tradition|custom)\b',
        r'\b(art|artistic|literature|literary|music|musical|creative)\b',
        r'\b(style|genre|movement|period|school|influence|inspiration)\b',
        
        # Entertainment/Gaming context
        r'\b(entertainment|game|gaming|player|audience|viewer|fan)\b',
        r'\b(story|narrative|plot|character|theme|genre|series)\b',
        r'\b(platform|console|developer|publisher|release|launch)\b',
        
        # Food/Dining context
        r'\b(cuisine|culinary|cooking|preparation|recipe|ingredient)\b',
        r'\b(restaurant|chef|kitchen|dining|meal|dish|flavor)\b',
        r'\b(nutrition|dietary|healthy|organic|traditional|regional)\b',
        
        # Health/Medical context
        r'\b(health|medical|clinical|therapeutic|treatment|diagnosis)\b',
        r'\b(patient|doctor|physician|hospital|healthcare|medicine)\b',
        r'\b(risk|safety|side\s+effect|benefit|harm|prevention)\b',
        
        # Legal/Crime context
        r'\b(legal|law|court|justice|criminal|crime|investigation)\b',
        r'\b(evidence|trial|case|prosecution|defense|penalty|sentence)\b',
        r'\b(judge|jury|attorney|lawyer|police|enforcement)\b',
        
        # Sports/Fitness context
        r'\b(sport|sports|athletic|fitness|training|exercise|competition)\b',
        r'\b(team|player|athlete|coach|championship|tournament|league)\b',
        r'\b(performance|skill|technique|strategy|physical|mental)\b',
        
        # Transportation context
        r'\b(transportation|transport|vehicle|travel|journey|trip)\b',
        r'\b(route|road|highway|infrastructure|traffic|system)\b',
        r'\b(passenger|cargo|freight|logistics|network|service)\b',
        
        # Fashion/Beauty context
        r'\b(fashion|style|design|designer|clothing|apparel|wear)\b',
        r'\b(beauty|cosmetic|makeup|skincare|hair|appearance)\b',
        r'\b(trend|seasonal|collection|brand|luxury|affordable)\b',
        
        # Technology/Software context
        r'\b(technology|software|hardware|computer|digital|electronic)\b',
        r'\b(development|programming|coding|system|platform|interface)\b',
        r'\b(user|developer|engineer|architect|database|network)\b',
        
        # Geographic/Environmental context
        r'\b(geographic|geographical|location|region|climate|environment)\b',
        r'\b(country|city|state|province|territory|area|zone)\b',
        r'\b(population|demographic|inhabitant|resident|citizen)\b',
        
        # Social/Relationship context (including adult content)
        r'\b(social|relationship|community|family|personal|intimate)\b',
        r'\b(interaction|communication|behavior|psychology|emotional)\b',
        r'\b(adult|mature|private|personal|individual|human)\b',
        
        # Home/Hobbies context
        r'\b(home|house|household|domestic|interior|exterior)\b',
        r'\b(hobby|craft|diy|project|collection|activity|leisure)\b',
        r'\b(garden|decoration|furniture|appliance|tool|equipment)\b',
        
        # Industrial context
        r'\b(industrial|manufacturing|production|factory|facility)\b',
        r'\b(machinery|equipment|process|operation|maintenance)\b',
        r'\b(worker|employee|safety|efficiency|automation)\b',
        
        # Relationships and connections (all domains)
        r'\b(related\s+to|connected\s+to|associated\s+with|part\s+of|member\s+of)\b',
        r'\b(influenced|affected|inspired|based\s+on|derived\s+from|adapted)\b',
        r'\b(leading\s+to|resulting\s+in|contributing\s+to|causing|producing)\b',
        
        # Detailed descriptions (all domains)
        r'\b(specifically|particularly|especially|notably|mainly|primarily)\b',
        r'\b(example|instance|case|illustration|demonstration|sample)\b',
        r'\b(details|information|evidence|data|facts|statistics|figures)\b',
        
        # Temporal context (all domains)
        r'\b(century|decade|year|period|era|age|epoch|time|timeline)\b',
        r'\b(historical|historic|ancient|medieval|renaissance|industrial)\b',
        r'\b(current|modern|contemporary|recent|latest|new|emerging)\b'
    ]
    
    context_matches = 0
    text_lower = text.lower()
    
    for pattern in context_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            context_matches += 1
    
    return context_matches >= 4  # At least 4 different types of contextual content

def check_unique_words(text: str, min_unique_words: int = 50) -> bool:
    """Check if article has sufficient vocabulary diversity"""
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = set(words)
    return len(unique_words) >= min_unique_words

def check_article_length(text: str, min_chars: int = 600) -> bool:
    """Check if article meets minimum length for substantial content"""
    return len(text.strip()) >= min_chars

def check_sentence_quality(text: str) -> bool:
    """Check for proper sentence structure and variety"""
    
    # Extract sentences
    sentences = re.split(r'[.!?]+', text)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    if len(valid_sentences) < 5:
        return False
    
    # Check sentence length variety (avoid repetitive short sentences)
    sentence_lengths = [len(s.split()) for s in valid_sentences]
    avg_length = sum(sentence_lengths) / len(sentence_lengths)
    
    # Good articles have average sentence length between 10-25 words
    if avg_length < 8 or avg_length > 30:
        return False
    
    # Check for sentence variety (some long, some short)
    long_sentences = [l for l in sentence_lengths if l > 15]
    return len(long_sentences) >= len(valid_sentences) * 0.3  # At least 30% long sentences

def check_content_depth_vs_structure(text: str) -> bool:
    """
    Check if article has substantial descriptive content vs just structural/factual content
    This helps filter out articles that are mostly results, dates, and basic facts
    """
    
    # Count structural/factual content (dates, numbers, short factual statements)
    structural_patterns = [
        r'\b\d{1,2}[,\s]+\d{4}\b',  # Dates
        r'\b\d+\s*(kg|m|km|cm|mm|lb|ft|inches?)\b',  # Measurements
        r'\b\d+[–-]\d+\b',  # Scores, ranges
        r'^\s*\w+:\s*\w+\s*$',  # Simple key-value pairs
        r'\b(first|second|third|1st|2nd|3rd|won|lost|defeated)\b',  # Competition results
    ]
    
    # Count descriptive/explanatory content
    descriptive_patterns = [
        r'\b(described\s+as|known\s+for|characterized\s+by|famous\s+for)\b',
        r'\b(because|since|due\s+to|as\s+a\s+result|therefore|however|although)\b',
        r'\b(style|approach|method|technique|process|system|way)\b',
        r'\b(influenced|inspired|affected|impact|effect|significance)\b',
        r'\b(developed|created|designed|built|established|formed)\b',
    ]
    
    text_lower = text.lower()
    sentences = re.split(r'[.!?]+', text)
    
    structural_sentences = 0
    descriptive_sentences = 0
    
    for sentence in sentences:
        if len(sentence.strip()) < 15:  # Skip very short fragments
            continue
        
        sentence_lower = sentence.lower()
        
        # Check if sentence is primarily structural/factual
        structural_matches = sum(1 for pattern in structural_patterns 
                               if re.search(pattern, sentence_lower))
        
        # Check if sentence is descriptive/explanatory
        descriptive_matches = sum(1 for pattern in descriptive_patterns 
                                if re.search(pattern, sentence_lower))
        
        if structural_matches > 0 and descriptive_matches == 0:
            structural_sentences += 1
        elif descriptive_matches > 0:
            descriptive_sentences += 1
    
    total_sentences = structural_sentences + descriptive_sentences
    
    if total_sentences == 0:
        return False
    
    # Article should have at least 40% descriptive content (not just facts/results)
    descriptive_ratio = descriptive_sentences / total_sentences
    return descriptive_ratio >= 0.4

def check_content_quality(title: str, text: str) -> Dict[str, bool]:
    """
    Domain-agnostic article quality check for TOT-relevant content
    """
    
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
    
    # Essential requirements: not a list page, sufficient length and vocabulary
    essential_checks = [
        checks['is_not_list_page'], 
        checks['has_min_unique_words'], 
        checks['has_min_length']
    ]
    
    # Content quality checks: explanatory, structured, contextual, well-written, substantial
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

def process_topic(task: Tuple[str, pd.DataFrame, str]) -> int:
    """
    Process a single topic, cleaning articles based on domain-agnostic quality criteria
    Optionally start processing from a given index.
    """
    topic_name, df, output_dir, start_idx = task
    cleaned_articles = []
    quality_stats = {
        'total_processed': 0,
        'low_quality_filtered': 0,
        'list_pages_filtered': 0,
        'quality_checks': {
            'is_not_list_page': 0,
            'has_min_unique_words': 0,
            'has_min_length': 0,
            'has_explanatory_content': 0,
            'has_coherent_paragraphs': 0,
            'has_contextual_information': 0,
            'has_good_sentences': 0,
            'has_sufficient_content_depth': 0
        }
    }
    
    logger.info(f"Processing topic: {topic_name} with {len(df)} articles, starting from idx={start_idx}.")
    
    # sort the df by title to ensure consistent processing order
    df = df.sort_values(by='title').reset_index(drop=True)
    for idx, row in df.iterrows():
        
        # skip until we reach the start index
        if idx < start_idx:
            continue
        
        # doc_id = '1154'
        doc_id = str(row['id'])
        quality_stats['total_processed'] += 1
        
        if doc_id not in doc_index:
            logger.warning(f"Document ID {doc_id} not found in index.")
            continue
        
        shard_path, row_index = doc_index[doc_id]
        article_df = pd.read_parquet(shard_path, engine='pyarrow', filters=[('id', '==', doc_id)])
        
        if article_df.empty:
            logger.warning(f"Article with ID {doc_id} is empty.")
            continue
        
        article_text = article_df.iloc[0]['text']
        article_title = article_df.iloc[0]['title']
            
        quality_checks = check_content_quality(article_title, article_text)
        
        # Update statistics
        for check_name, result in quality_checks.items():
            if check_name != 'overall_quality' and result:
                quality_stats['quality_checks'][check_name] += 1
        
        # Track list pages separately
        if not quality_checks['is_not_list_page']:
            quality_stats['list_pages_filtered'] += 1
        
        if not quality_checks['overall_quality']:
            logger.debug(f"Article '{article_title}' (ID: {doc_id}) failed quality checks: {quality_checks}")
            quality_stats['low_quality_filtered'] += 1
            continue
            
        cleaned_articles.append({
            'id': doc_id,
            'title': article_title,
            'url': article_df['url'].values[0],
            'text': article_text,
            'topic': topic_name,
            'confidence': row['confidence']
        })
        
        # Log progress every 1000 articles, and save the cleaned articles periodically
        if idx > 0 and idx % 1000 == 0:
            logger.info(f"Processed {idx} articles for topic '{topic_name}' out of {len(df)}...")

        # Save intermediate results
        save_interval = 1000  # Save every 10,000 articles
        if len(cleaned_articles) % save_interval == 0:
            logger.info(f"Saving intermediate cleaned articles for topic '{topic_name}' at index {idx}...")
            # Save to a temporary file to avoid memory issues
            if cleaned_articles:
                intermediate_file = os.path.join(output_dir, f"{topic_name}_cleaned_{idx}.parquet")
                pd.DataFrame(cleaned_articles).to_parquet(intermediate_file, index=False)
            logger.info(f"Saved {len(cleaned_articles)} articles to {intermediate_file}")
            cleaned_articles = []  # Reset for next batch

    # Save the final cleaned articles for this topic
    if cleaned_articles:
        final_file = os.path.join(output_dir, f"{topic_name}_cleaned_last.parquet")
        pd.DataFrame(cleaned_articles).to_parquet(final_file, index=False)
        logger.info(f"Saved final cleaned articles for topic '{topic_name}' to {final_file}")

    logger.info(f"Domain-agnostic quality statistics for topic '{topic_name}':")
    logger.info(f"  Total processed: {quality_stats['total_processed']}")
    logger.info(f"  List/disambiguation pages filtered: {quality_stats['list_pages_filtered']}")
    logger.info(f"  Other low quality filtered: {quality_stats['low_quality_filtered'] - quality_stats['list_pages_filtered']}")
    logger.info(f"  Kept: {len(cleaned_articles)}")
    logger.info(f"  Individual check pass rates:")
    for check_name, count in quality_stats['quality_checks'].items():
        rate = (count / quality_stats['total_processed']) * 100 if quality_stats['total_processed'] > 0 else 0
        logger.info(f"    {check_name}: {count}/{quality_stats['total_processed']} ({rate:.1f}%)")

    return len(cleaned_articles)

if __name__ == "__main__":
    parquet_shards_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/split_parquet_shards/"
    topic_csv_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/topic_grouped_csv"
    output_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/cleaned_articles_parquet"

    os.makedirs(output_dir, exist_ok=True)

    global doc_index
    logger.info("Building article index...")
    doc_index = build_shard_index(parquet_shards_dir)

    logger.info("Loading topic CSV files...")
    topic_data = load_topic_csv_files(topic_csv_dir)

    # lets try "home_hobbies" as an example topic
    # topic_data = {k: v for k, v in topic_data.items() if k == 'home_hobbies'}
    # process_topic(("home_hobbies", topic_data['home_hobbies'], output_dir))
    

    # Process all topics in parallel
    temp_tasks = [(name, df, output_dir) for name, df in topic_data.items()]
    tasks = []
    
    for topic_name, df in topic_data.items():
        if topic_name == 'adult_content':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'art_design':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'crime_law':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'education_jobs':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'electronics_hardware':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'entertainment':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'fashion_beauty':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'finance_business':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'food_dining':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'games':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'health':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'history_geography':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'home_hobbies':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'industrial':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'literature':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'politics':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'religion':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'science_math_technology':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'social_life':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'software_development':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'software':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'sports_fitness':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'transportation':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
        if topic_name == 'travel_tourism':
            start_idx = 0
            tasks.append((topic_name, df, output_dir, start_idx))
    
    logger.info(f"Starting parallel processing of {len(tasks)} topics...")
    start = time.time()
    
    with Pool(cpu_count()) as pool:
        results = pool.map(process_topic, tasks)
    
    end = time.time()
    logger.info(f"Finished cleaning all topics in {end - start:.2f} seconds")
    logger.info(f"Total cleaned articles across all topics: {sum(results)}")
    
    # Print summary by topic
    for i, (topic_name, _, _) in enumerate(tasks):
        logger.info(f"  {topic_name}: {results[i]} good articles out of {len(topic_data[topic_name])} total")
    logger.info("Article cleaning completed successfully.")