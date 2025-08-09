#!/usr/bin/env python3
"""
Wikipedia Pageview Aggregator

This script aggregates English Wikipedia page view counts over the period of 
2022-nov-01 to 2023-oct-31 from Wikimedia dumps.

Features:
- Downloads monthly user pageview data from Wikimedia dumps
- Filters for English Wikipedia pages only
- Aggregates view counts across the time period
- Saves results as TSV with page ID, title, and view count
- Resumable process with checkpoint files
- Automatic cleanup of downloaded files to save storage
"""

import os
import sys
import json
import bz2
import csv
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wikipedia_pageview_aggregator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class WikipediaPageviewAggregator:
    def __init__(self, data_dir: str = "tmp", output_dir: str = "outputs/page_view", checkpoint_file: str = "aggregation_checkpoint.json"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / checkpoint_file
        self.base_url = "https://dumps.wikimedia.org/other/pageview_complete/monthly"
        
        # Page view aggregation data
        self.page_views = defaultdict(int)  # page_id -> total_views
        self.processed_months = set()
        
        # Load existing checkpoint if available
        self.load_checkpoint()
    
    def generate_month_urls(self, start_date: str, end_date: str) -> list:
        """Generate URLs for monthly pageview data between start and end dates."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        urls = []
        current = start.replace(day=1)  # Start from first day of month
        
        while current <= end:
            year = current.year
            month = current.month
            
            # Format: pageviews-YYYYMM-user.bz2
            filename = f"pageviews-{year:04d}{month:02d}-user.bz2"
            url = f"{self.base_url}/{year}/{year}-{month:02d}/{filename}"
            urls.append((url, filename, f"{year:04d}-{month:02d}"))
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return urls
    
    def download_file(self, url: str, filename: str) -> bool:
        """Download a file with progress tracking."""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"File {filename} already exists, skipping download")
            return True
        
        try:
            logger.info(f"Downloading {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end='', flush=True)
            
            print()  # New line after progress
            logger.info(f"Downloaded {filename} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def process_pageview_file(self, filename: str, month_id: str) -> bool:
        """Process a single pageview BZ2 file and aggregate English Wikipedia data."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.error(f"File {filename} does not exist")
            return False
        
        if month_id in self.processed_months:
            logger.info(f"Month {month_id} already processed, skipping")
            return True
        
        logger.info(f"Processing {filename}")
        
        try:
            processed_lines = 0
            english_wiki_lines = 0
            id_null_cnt = 0
            title_null_cnt = 0
            
            with bz2.open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    processed_lines += 1
                    
                    if processed_lines % 10000000 == 0:
                        logger.info(f"Processed {processed_lines} lines, found {english_wiki_lines} English Wikipedia entries")
                    
                    try:
                        # Check if line starts with 'en.wikipedia' and print debug info
                        if not line.strip().startswith('en.wikipedia'):
                            continue

                        # Parse line
                        parts = line.strip().split(' ')
                        # Continue with normal parsing
                        if len(parts) < 5:
                            continue
                        
                        domain_code = parts[0]

                        # Filter for English Wikipedia only
                        if domain_code == 'en.wikipedia':
                            page_title = parts[1]
                            id = parts[2]
                            count_views = int(parts[4])
                            try:
                                if id == 'null' or id == '-':
                                    id_null_cnt += 1
                                    continue
                                if page_title == 'null' or page_title == '-':
                                    title_null_cnt += 1
                                    continue

                                self.page_views[id] += count_views
                                english_wiki_lines += 1
                            except Exception as decode_error:
                                logger.warning(f"Failed to decode title {page_title}: {decode_error}")
                                continue
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse line: {line.strip()[:100]}... Error: {e}")
                        continue
            
            logger.info(f"Finished processing {filename}: {processed_lines} total lines, {english_wiki_lines} English Wikipedia entries")
            logger.info(f"Skipped {id_null_cnt} entries with null IDs, {title_null_cnt} entries with null/missing titles")
            
            # Mark month as processed
            self.processed_months.add(month_id)
            
            # Save checkpoint after processing each file
            self.save_checkpoint()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return False
    
    def cleanup_file(self, filename: str):
        """Delete downloaded file to save storage space."""
        filepath = self.data_dir / filename
        if filepath.exists():
            try:
                filepath.unlink()
                logger.info(f"Deleted {filename} to save storage")
            except Exception as e:
                logger.warning(f"Failed to delete {filename}: {e}")
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        checkpoint_data = {
            'processed_months': list(self.processed_months),
            'page_views': dict(self.page_views),
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Checkpoint saved with {len(self.page_views)} pages")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self):
        """Load progress from checkpoint file if it exists."""
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint file found, starting fresh")
            return
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            self.processed_months = set(checkpoint_data.get('processed_months', []))
            self.page_views = defaultdict(int, checkpoint_data.get('page_views', {}))
            
            logger.info(f"Checkpoint loaded: {len(self.processed_months)} months processed, "
                       f"{len(self.page_views)} pages in aggregation")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh")
    
    def save_results(self, output_file: str = "wikipedia_pageviews_2022_2023.tsv"):
        """Save aggregated results to TSV file."""
        output_path = self.output_dir / output_file
        
        logger.info(f"Saving results to {output_path}")
        
        # Sort pages by view count (descending)
        sorted_pages = sorted(self.page_views.items(), key=lambda x: x[1], reverse=True)
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                
                # Write header
                writer.writerow(['page_id', 'total_pageviews'])
                
                # Write data
                for page_id, total_views in sorted_pages:
                    writer.writerow([page_id, total_views])
            
            logger.info(f"Results saved: {len(sorted_pages)} pages written to {output_path}")
            
            # Print top 10 pages
            logger.info("Top 10 most viewed pages:")
            for i, (page_id, views) in enumerate(sorted_pages[:10], 1):
                logger.info(f"{i:2d}. Page ID {page_id}: {views:,} views")
                
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def run_aggregation(self, start_date: str = "2022-11-01", end_date: str = "2023-10-01"):
        """Run the complete aggregation process."""
        logger.info(f"Starting Wikipedia pageview aggregation from {start_date} to {end_date}")
        
        # Generate URLs for the time period
        urls = self.generate_month_urls(start_date, end_date)
        logger.info(f"Will process {len(urls)} months of data")
        
        total_months = len(urls)
        completed_months = 0
        
        for url, filename, month_id in urls:
            logger.info(f"Processing month {month_id} ({completed_months + 1}/{total_months})")
            
            # Skip if already processed
            if month_id in self.processed_months:
                logger.info(f"Month {month_id} already processed, skipping")
                completed_months += 1
                continue
            
            # Download file
            if not self.download_file(url, filename):
                logger.error(f"Failed to download {filename}, skipping")
                continue
            
            # Process file
            if self.process_pageview_file(filename, month_id):
                completed_months += 1
                logger.info(f"Successfully processed {month_id} ({completed_months}/{total_months})")
            else:
                logger.error(f"Failed to process {filename}")
            
            # Cleanup downloaded file to save space
            self.cleanup_file(filename)
            
            # Brief pause between downloads to be respectful to the server
            time.sleep(1)
        
        logger.info(f"Aggregation complete! Processed {completed_months}/{total_months} months")
        logger.info(f"Total unique pages: {len(self.page_views)}")
        logger.info(f"Total page views: {sum(self.page_views.values()):,}")
        
        # Save final results
        self.save_results()


def main():
    """Main function to run the aggregation."""
    aggregator = WikipediaPageviewAggregator()
    
    try:
        # Run aggregation for the specified time period
        aggregator.run_aggregation("2022-11-01", "2023-10-01")
        # aggregator.run_aggregation("2022-11-01", "2022-11-01")
        logger.info("Wikipedia pageview aggregation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Progress has been saved to checkpoint file.")
        logger.info("You can resume the process by running the script again.")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
 