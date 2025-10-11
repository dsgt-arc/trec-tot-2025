#!/usr/bin/env python3
"""
Wikipedia Article Sampling Script

This script samples articles from the wikipedia_articles database based on:
1. Word count percentiles (20%, 25%, 50%, 75%, 90%)
2. Page view percentiles (20%, 50%, 70%, 80%)
3. Category-based sampling with filtering criteria
4. Exports sampled article IDs to CSV
"""

import sqlite3
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple
from wikipedia_db import WikipediaDB

class WikipediaSampler:
    def __init__(self, db_path="outputs/wikipedia_data.db"):
        self.db = WikipediaDB(db_path)
        self.word_count_cutoffs = {}
        self.page_view_cutoffs = {}
        
    def calculate_word_count_cutoffs(self) -> Dict[str, int]:
        """Calculate cutoff values for word count percentiles (20%, 25%, 50%, 75%, 90%)"""
        conn = self.db.connect()
        cursor = conn.cursor()
        
        try:
            # Get all word counts, excluding NULL values
            cursor.execute('''
                SELECT word_count 
                FROM wikipedia_articles 
                WHERE word_count IS NOT NULL 
                ORDER BY word_count
            ''')
            
            word_counts = [row[0] for row in cursor.fetchall()]
            total_count = len(word_counts)
            
            if total_count == 0:
                print("WARNING: No articles with word count found")
                return {}
            
            percentiles = [20, 25, 50, 75, 90]
            cutoffs = {}
            
            for p in percentiles:
                index = int((p / 100) * total_count)
                if index >= total_count:
                    index = total_count - 1
                cutoffs[f'{p}%'] = word_counts[index]
            
            self.word_count_cutoffs = cutoffs
            print(f"Word count cutoffs: {cutoffs}")
            return cutoffs
            
        finally:
            conn.close()
    
    def calculate_page_view_cutoffs(self) -> Dict[str, int]:
        """Calculate cutoff values for page view percentiles (20%, 50%, 70%, 80%)"""
        conn = self.db.connect()
        cursor = conn.cursor()
        
        try:
            # Get all page views, excluding NULL values
            cursor.execute('''
                SELECT page_views 
                FROM wikipedia_articles 
                WHERE page_views IS NOT NULL 
                ORDER BY page_views
            ''')
            
            page_views = [row[0] for row in cursor.fetchall()]
            total_count = len(page_views)
            
            if total_count == 0:
                print("WARNING: No articles with page views found")
                return {}
            
            percentiles = [20, 50, 70, 80]
            cutoffs = {}
            
            for p in percentiles:
                index = int((p / 100) * total_count)
                if index >= total_count:
                    index = total_count - 1
                cutoffs[f'{p}%'] = page_views[index]
            
            self.page_view_cutoffs = cutoffs
            print(f"Page view cutoffs: {cutoffs}")
            return cutoffs
            
        finally:
            conn.close()
    
    def get_category_counts(self) -> Dict[str, int]:
        """Get count of articles per category (infobox_type)"""
        conn = self.db.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT infobox_type, COUNT(*) as count
                FROM wikipedia_articles 
                WHERE infobox_type IS NOT NULL
                GROUP BY infobox_type
                ORDER BY count DESC
            ''')
            
            categories = dict(cursor.fetchall())
            print(f"Found {len(categories)} categories")
            
            # Show category distribution
            for category, count in list(categories.items())[:10]:
                print(f"   - {category}: {count} articles")
            
            return categories
            
        finally:
            conn.close()
    
    def determine_sample_size(self, category_count: int) -> int:
        """Determine sample size based on category count"""
        if category_count < 10:
            return 0  # Skip categories with < 10 articles
        elif 10 <= category_count < 100:
            return random.choice([1, 2])
        elif 100 <= category_count < 500:
            return 3
        elif 500 <= category_count < 1000:
            return 4
        else:  # >= 1000
            return 5
    
    def get_filtering_criteria(self) -> Dict[str, int]:
        """
        Calculate and return filtering criteria thresholds
        Returns dict with min_word_count, max_word_count, min_page_views
        """
        # Ensure we have cutoffs calculated
        if not self.word_count_cutoffs:
            self.calculate_word_count_cutoffs()
        if not self.page_view_cutoffs:
            self.calculate_page_view_cutoffs()
        
        # Get minimum thresholds (25% word count, 80% page views)
        min_word_count = self.word_count_cutoffs.get('25%', 0)
        max_word_count = 5000  # Maximum word count limit
        min_page_views = self.page_view_cutoffs.get('80%', 0)
        
        criteria = {
            'min_word_count': min_word_count,
            'max_word_count': max_word_count,
            'min_page_views': min_page_views
        }
        
        print(f"Filtering criteria:")
        print(f"   - Word count range: {min_word_count} - {max_word_count}")
        print(f"   - Minimum page views (80%): {min_page_views}")
        
        # Count total articles that meet the filtering criteria
        conn = self.db.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT COUNT(*) 
                FROM wikipedia_articles 
                WHERE word_count >= ?
                  AND word_count <= ?
                  AND page_views >= ?
                  AND word_count IS NOT NULL 
                  AND page_views IS NOT NULL
                  AND infobox_type IS NOT NULL
            ''', (min_word_count, max_word_count, min_page_views))
            
            eligible_count = cursor.fetchone()[0]
            print(f"   - Total articles meeting criteria: {eligible_count}")
            criteria['total_eligible'] = eligible_count
            
        finally:
            conn.close()
        
        return criteria

    def get_top_bottom_page_views(self, top_n: int = 20, bottom_n: int = 20) -> Dict[str, List[Tuple]]:
        """
        Get the top N and bottom N pages by page views
        
        Args:
            top_n: Number of top pages to retrieve (default: 20)
            bottom_n: Number of bottom pages to retrieve (default: 20)
            
        Returns:
            Dict with 'top' and 'bottom' keys containing lists of tuples:
            (wikipedia_id, title, page_views, word_count, infobox_type)
        """
        conn = self.db.connect()
        cursor = conn.cursor()
        
        result = {'top': [], 'bottom': []}
        
        try:
            # Get top N pages with highest page views
            print(f"Getting top {top_n} pages by page views...")
            cursor.execute('''
                SELECT wikipedia_id, title, page_views, word_count, infobox_type
                FROM wikipedia_articles 
                WHERE page_views IS NOT NULL
                ORDER BY page_views DESC
                LIMIT ?
            ''', (top_n,))
            
            result['top'] = cursor.fetchall()
            
            # Get bottom N pages with lowest page views (excluding 0 views)
            print(f"Getting bottom {bottom_n} pages by page views...")
            cursor.execute('''
                SELECT wikipedia_id, title, page_views, word_count, infobox_type
                FROM wikipedia_articles 
                WHERE page_views IS NOT NULL AND page_views > 0
                ORDER BY page_views ASC
                LIMIT ?
            ''', (bottom_n,))
            
            result['bottom'] = cursor.fetchall()
            
            # Print summary
            print(f"\nTop {len(result['top'])} pages by page views:")
            for i, (wiki_id, title, views, word_count, category) in enumerate(result['top'], 1):
                print(f"   {i:2d}. {title[:50]:<50} | ID: {wiki_id} | Views: {views:,} | Words: {word_count or 'N/A'}")
            
            print(f"\nBottom {len(result['bottom'])} pages by page views (excluding 0 views):")
            for i, (wiki_id, title, views, word_count, category) in enumerate(result['bottom'], 1):
                print(f"   {i:2d}. {title[:50]:<50} | ID: {wiki_id} | Views: {views:,} | Words: {word_count or 'N/A'}")
            
            return result
            
        finally:
            conn.close()

    def sample_articles_by_category(self, test_mode: bool = False, test_limit: int = 10) -> List[Tuple[str, str, str]]:
        """
        Sample articles by category with filtering criteria
        
        Args:
            test_mode: If True, limits sampling to first test_limit categories
            test_limit: Number of categories to process in test mode (default: 10)
            
        Returns list of tuples: (wikipedia_id, title, infobox_type)
        """
        # Get filtering criteria
        criteria = self.get_filtering_criteria()
        min_word_count = criteria['min_word_count']
        max_word_count = criteria['max_word_count']
        min_page_views = criteria['min_page_views']
        
        category_counts = self.get_category_counts()
        
        if test_mode:
            # Limit to first N categories for testing
            category_items = list(category_counts.items())[:test_limit]
            category_counts = dict(category_items)
            print(f"\nTEST MODE: Processing first {len(category_counts)} categories out of {len(category_items) if len(category_items) < test_limit else test_limit}")
        
        sampled_articles = []
        
        conn = self.db.connect()
        cursor = conn.cursor()
        
        try:
            for category, total_count in category_counts.items():
                sample_size = self.determine_sample_size(total_count)
                
                if sample_size == 0:
                    continue
                
                print(f"\nSampling {sample_size} articles from '{category}' ({total_count} total)")
                
                # First, count how many articles in this category meet the criteria
                cursor.execute('''
                    SELECT COUNT(*)
                    FROM wikipedia_articles 
                    WHERE infobox_type = ?
                      AND word_count >= ?
                      AND word_count <= ?
                      AND page_views >= ?
                      AND word_count IS NOT NULL 
                      AND page_views IS NOT NULL
                ''', (category, min_word_count, max_word_count, min_page_views))
                
                eligible_in_category = cursor.fetchone()[0]
                print(f"   - Articles meeting criteria: {eligible_in_category}/{total_count}")
                
                if eligible_in_category == 0:
                    print(f"   - Skipping: No articles meet filtering criteria")
                    continue
                
                # Get eligible articles from this category
                cursor.execute('''
                    SELECT wikipedia_id, title, infobox_type
                    FROM wikipedia_articles 
                    WHERE infobox_type = ?
                      AND word_count >= ?
                      AND word_count <= ?
                      AND page_views >= ?
                      AND word_count IS NOT NULL 
                      AND page_views IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT ?
                ''', (category, min_word_count, max_word_count, min_page_views, sample_size))
                
                category_samples = cursor.fetchall()
                sampled_articles.extend(category_samples)
                
                print(f"   Sampled {len(category_samples)} articles")
                # for article in category_samples:
                #     print(f"      - {article[0]}: {article[1]}")
        
        finally:
            conn.close()
        
        print(f"\nTotal sampled articles: {len(sampled_articles)}")
        return sampled_articles
    
    def save_to_csv(self, articles: List[Tuple[str, str, str]], output_file: str = "outputs/sampled_wikipedia_articles.csv"):
        """Save sampled articles to CSV file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['wikipedia_id', 'title', 'infobox_type'])
            
            # Write articles
            for article in articles:
                writer.writerow(article)
        
        print(f"Saved {len(articles)} sampled articles to: {output_path}")
        return output_path
    
    def generate_sampling_report(self, test_mode: bool = False, test_limit: int = 10):
        """
        Generate a comprehensive sampling report
        
        Args:
            test_mode: If True, limits sampling to first test_limit categories
            test_limit: Number of categories to process in test mode (default: 10)
        """
        print("=" * 60)
        if test_mode:
            print("WIKIPEDIA ARTICLE SAMPLING REPORT (TEST MODE)")
        else:
            print("WIKIPEDIA ARTICLE SAMPLING REPORT")
        print("=" * 60)
        
        # Step 1: Calculate cutoffs
        print("\n1. Calculating Word Count Cutoffs...")
        word_cutoffs = self.calculate_word_count_cutoffs()
        
        print("\n2. Calculating Page View Cutoffs...")
        page_cutoffs = self.calculate_page_view_cutoffs()
        
        # Step 3: Sample articles
        print("\n3. Sampling Articles by Category...")
        sampled_articles = self.sample_articles_by_category(test_mode=test_mode, test_limit=test_limit)
        
        # Step 4: Save to CSV
        print("\n4. Saving Results...")
        output_file = self.save_to_csv(sampled_articles)
        
        print("\n" + "=" * 60)
        print("SAMPLING COMPLETE")
        print("=" * 60)
        print(f"Output file: {output_file}")
        print(f"Total articles sampled: {len(sampled_articles)}")
        
        return output_file

def main():
    """Main function to run the sampling process"""
    import sys
    
    # Check if test mode is requested
    test_mode = '--test' in sys.argv or '--test-mode' in sys.argv
    
    if test_mode:
        print("Starting Wikipedia Article Sampling (TEST MODE - First 10 categories)...")
    else:
        print("Starting Wikipedia Article Sampling...")
    
    # Initialize sampler
    sampler = WikipediaSampler()
    
    # Generate sampling report
    output_file = sampler.generate_sampling_report(test_mode=test_mode)
    
    if test_mode:
        print(f"\nTest sampling completed successfully!")
        print("To run full sampling, use: python run_sampling.py")
    else:
        print(f"\nSampling completed successfully!")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
