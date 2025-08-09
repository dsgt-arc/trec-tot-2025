
import logging
import sys
import bz2
from pathlib import Path
from collections import defaultdict
import json


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wikipedia_pageview_aggregator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def process_pageview_file() -> bool:
    """Process a single pageview BZ2 file and aggregate English Wikipedia data."""
    filepath = Path('/home/wenxin/project/data/pageview/pageviews-202211-user.bz2')
    page_views = defaultdict(int)
    page_name = {}

    if not filepath.exists():
        logger.error(f"File {filepath} does not exist")
        return False

    logger.info(f"Processing {filepath}")

    try:
        processed_lines = 0
        english_wiki_lines = 0
        id_null_cnt = 0
        title_null_cnt = 0
        
        with bz2.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                processed_lines += 1
                
                if processed_lines % 1000000 == 0:
                    logger.info(f"Processed {processed_lines} lines, found {english_wiki_lines} English Wikipedia entries")
                
                try:
                    
                    # Parse line first
                    parts = line.strip().split(' ')

                    # Check if line starts with 'en.wikipedia' and print debug info
                    if not line.strip().startswith('en.wikipedia'):
                        continue
                    
                    # Continue with normal parsing
                    if len(parts) < 5:
                        continue
                    
                    domain_code = parts[0]
                    page_title = parts[1]
                    id = parts[2]
                    count_views = int(parts[4])

                    # Filter for English Wikipedia only
                    if domain_code == 'en.wikipedia':
                        # print(f"Line {processed_lines}, {line}")
                        # Decode URL-encoded page titles
                        try:
                            if id == 'null':
                                id_null_cnt += 1
                                continue
                            if page_title == 'null' or page_title == '-':
                                title_null_cnt += 1
                                continue

                            page_views[id] += count_views
                            if id not in page_name:
                                page_name[id] = {page_title}
                            else:
                                page_name[id].add(page_title)
                            english_wiki_lines += 1
                        except Exception as decode_error:
                            logger.warning(f"Failed to decode title {page_title}: {decode_error}")
                            continue
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse line: {line.strip()[:100]}... Error: {e}")
                    continue
        
        logger.info(f"Finished processing {filepath}: {processed_lines} total lines, {english_wiki_lines} English Wikipedia entries")
        # print the number of unique page views
        logger.info(f"Unique page views: {len(page_views)}")
        logger.info(f"Entries with null IDs: {id_null_cnt}, Entries with null or invalid titles: {title_null_cnt}")

        # save page_views and page_name
        with open('page_views.json', 'w') as f:
            json.dump(page_views, f, indent=2)
        
        # Convert sets to lists before saving to JSON
        page_name_serializable = {k: list(v) for k, v in page_name.items()}
        with open('page_name.json', 'w') as f:
            json.dump(page_name_serializable, f, indent=2)

        return True
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return False

if __name__ == "__main__":
    process_pageview_file()
