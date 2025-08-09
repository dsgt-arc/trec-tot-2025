#!/usr/bin/env python3
"""
Database operations helper for Wikipedia data
"""
import sqlite3
import csv
from pathlib import Path
from typing import List, Dict, Optional

class WikipediaDB:
    def __init__(self, db_path="outputs/wikipedia_data.db"):
        self.db_path = db_path
        
    def connect(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def insert_article(self, wikipedia_id: str, title: str, 
                      word_count: int = None, page_views: int = None, 
                      offset_start: int = None, offset_end: int = None,
                      infobox_type: str = None):
        """Insert or update a Wikipedia article"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO wikipedia_articles 
                (wikipedia_id, title, word_count, page_views, offset_start, offset_end, infobox_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (wikipedia_id, title, word_count, page_views, offset_start, offset_end, infobox_type))
            
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def get_article(self, wikipedia_id: str) -> Optional[Dict]:
        """Get article by Wikipedia ID"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT wikipedia_id, title, word_count, page_views, offset_start, offset_end, infobox_type, created_at, updated_at
                FROM wikipedia_articles 
                WHERE wikipedia_id = ?
            ''', (wikipedia_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'wikipedia_id': row[0],
                    'title': row[1],
                    'word_count': row[2],
                    'page_views': row[3],
                    'offset_start': row[4],
                    'offset_end': row[5],
                    'infobox_type': row[6],
                    'created_at': row[7],
                    'updated_at': row[8]
                }
            return None
        finally:
            conn.close()
    
    def search_articles(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search articles by title or category"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT wikipedia_id, title, word_count, page_views, offset_start, offset_end, infobox_type
                FROM wikipedia_articles 
                WHERE title LIKE ?
                ORDER BY page_views DESC
                LIMIT ?
            ''', (f'%{search_term}%', limit))
            
            rows = cursor.fetchall()
            return [
                {
                    'wikipedia_id': row[0],
                    'title': row[1],
                    'word_count': row[2],
                    'page_views': row[3],
                    'offset_start': row[4],
                    'offset_end': row[5],
                    'infobox_type': row[6]
                }
                for row in rows
            ]
        finally:
            conn.close()
   
    def get_stats(self):
        """Get database statistics"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Count articles
            cursor.execute("SELECT COUNT(*) FROM wikipedia_articles")
            article_count = cursor.fetchone()[0]
            
            # Top infobox types
            cursor.execute('''
                SELECT infobox_type, COUNT(*) as count 
                FROM wikipedia_articles 
                WHERE infobox_type IS NOT NULL 
                GROUP BY infobox_type 
                ORDER BY count DESC 
                LIMIT 5
            ''')
            top_infobox_types = cursor.fetchall()
            
            # Articles with most views
            cursor.execute('''
                SELECT title, page_views 
                FROM wikipedia_articles 
                WHERE page_views IS NOT NULL 
                ORDER BY page_views DESC 
                LIMIT 5
            ''')
            top_viewed = cursor.fetchall()
            
            return {
                'total_articles': article_count,
                'top_infobox_types': top_infobox_types,
                'most_viewed_articles': top_viewed
            }
        finally:
            conn.close()

# Example usage
if __name__ == "__main__":
    db = WikipediaDB()
    
    # Test basic operations
    print("🧪 Testing database operations...")
    
    # Insert test article
    db.insert_article("test123", "Test Article", 1000, 500, infobox_type="test_type")
    
    # Retrieve article
    article = db.get_article("test123")
    print(f"📄 Retrieved article: {article['title']} (infobox: {article['infobox_type']})")
    
    # Get stats
    stats = db.get_stats()
    print(f"📊 Database stats: {stats['total_articles']} articles")
