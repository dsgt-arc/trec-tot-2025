#!/usr/bin/env python3
"""
Database setup script for Wikipedia data
Creates SQLite database with tables for articles and queries
"""
import sqlite3
from pathlib import Path

def setup_database(db_path="wikipedia_data.db"):
    """Create SQLite database with Wikipedia tables"""
    
    # Create database directory if it doesn't exist
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to SQLite database (creates file if doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create wikipedia_articles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wikipedia_articles (
                id INTEGER PRIMARY KEY,
                wikipedia_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                word_count INTEGER,
                page_views INTEGER,
                offset_start INTEGER,
                offset_end INTEGER,
                infobox_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_articles_wikipedia_id 
            ON wikipedia_articles (wikipedia_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_articles_title 
            ON wikipedia_articles (title)
        ''')
        
        conn.commit()
        print(f"✅ Database created successfully: {db_path}")
        print(f"📊 Tables created:")
        print(f"   - wikipedia_articles (id, wikipedia_id, title, word_count, page_views, offset_start, offset_end, infobox_type)")
        
        # Show table info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"\n📋 Database contains {len(tables)} tables: {[t[0] for t in tables]}")
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        conn.rollback()
    finally:
        conn.close()
