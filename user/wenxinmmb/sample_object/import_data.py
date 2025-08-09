#!/usr/bin/env python3
"""
Import Wikipedia data from multiple sources into SQLite database
"""
from wikipedia_db import WikipediaDB
import csv
import os
import json

def import_article_metadata():
    """Import data from multiple sources: corpus, offset mapping, word count, and pageview files"""
    
    # Define file paths
    corpus_file = "/home/wenxin/project/data/2025/corpus.jsonl"
    offset_file = "/home/wenxin/project/data/2025/corpus-offset-mapping.json"
    word_count_file = "outputs/word_count_analysis.csv"
    pageview_file = "outputs/page_view/wikipedia_pageviews_2022_2023-v1.tsv"
    infobox_file = "outputs/infobox/infobox_per_article.tsv"
    
    # Check if all files exist
    missing_files = []
    for file_path, name in [
        (corpus_file, "corpus file"),
        (offset_file, "offset mapping file"),
        (word_count_file, "word count file"), 
        (pageview_file, "pageview file"),
        (infobox_file, "infobox file")
    ]:
        if not os.path.exists(file_path):
            missing_files.append(f"{name}: {file_path}")
    
    if missing_files:
        print(f"❌ Missing files:")
        for missing in missing_files:
            print(f"   {missing}")
        return
    
    print(f"📥 Importing data from multiple sources...")
    
    # Initialize database
    db = WikipediaDB()
    
    # Step 1: Load offset mapping data
    print(f"Loading offset mapping from {offset_file}...")
    with open(offset_file, 'r', encoding='utf-8') as f:
        offset_mapping = json.load(f)
    print(f"✅ Loaded offset mapping for {len(offset_mapping)} articles")

    # Step 2: Load word count data and merge into offset mapping
    print(f"Loading word count data from {word_count_file}...")
    with open(word_count_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            article_id = row.get('id')
            word_count = row.get('word_count')
            if article_id and word_count and article_id in offset_mapping:
                offset_mapping[article_id]['wc'] = int(word_count)
    
    wc_count = sum(1 for v in offset_mapping.values() if 'wc' in v)
    print(f"✅ Merged word counts for {wc_count} articles")
    
    # Step 3: Load pageview data and merge into offset mapping
    print(f"Loading pageview data from {pageview_file}...")
    with open(pageview_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        # Skip header if exists
        first_row = next(csv_reader, None)
        print(f"Header: {first_row}")

        # Process remaining rows
        for row in csv_reader:
            assert len(row) >= 3, f"Row must have at least 3 columns: {row}"
            article_id = row[0]
            # Only add to offset_mapping if article_id exists in offset_mapping
            if article_id in offset_mapping:
                offset_mapping[article_id]['pvc'] = int(row[2])
    
    pvc_count = sum(1 for v in offset_mapping.values() if 'pvc' in v)
    print(f"✅ Merged pageview data for {pvc_count} articles")
    
    # Step 4: Load infobox data and merge into offset mapping
    print(f"Loading infobox data from {infobox_file}...")
    with open(infobox_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        # Skip header if exists
        first_row = next(csv_reader, None)
        print(f"Header: {first_row}")

        # Process remaining rows
        for row in csv_reader:
            if len(row) >= 3:
                article_id = row[0]
                infobox_template = row[2]
                # Only add to offset_mapping if article_id exists in offset_mapping
                if article_id in offset_mapping:
                    offset_mapping[article_id]['infobox_type'] = infobox_template
    
    infobox_count = sum(1 for v in offset_mapping.values() if 'infobox_type' in v)
    print(f"✅ Merged infobox data for {infobox_count} articles")
    
    # Show consolidated data summary
    print(f"📊 Consolidated data summary:")
    print(f"   Total articles with offsets: {len(offset_mapping)}")
    print(f"   Articles with word counts: {wc_count}")
    print(f"   Articles with page views: {pvc_count}")
    print(f"   Articles with infobox types: {infobox_count}")
    complete_count = sum(1 for v in offset_mapping.values() if 'wc' in v and 'pvc' in v and 'infobox_type' in v)
    print(f"   Articles with complete data: {complete_count}")
    
    # Step 5: Process corpus file and combine all data
    print(f"🚀 Processing corpus file and combining data...")
    processed_count = 0
    errors = []
    
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    article = json.loads(line.strip())
                    article_id = article.get('id')
                    title = article.get('title')
                    
                    if not article_id or not title:
                        errors.append(f"Line {line_num}: Missing id or title")
                        continue
                    
                    # Skip if article_id is not in offset_mapping (our master list)
                    if article_id not in offset_mapping:
                        errors.append(f"Line {line_num}: Article ID '{article_id}' not found in offset mapping")
                        continue
                    
                    # Get data from consolidated mapping
                    article_data = offset_mapping.get(article_id, {})
                    offset_start = article_data.get('offset_start')
                    offset_end = article_data.get('offset_end')
                    word_count = article_data.get('wc')
                    page_views = article_data.get('pvc')
                    infobox_type = article_data.get('infobox_type')
                    if page_views is None:
                        page_views = 0  # Default to 0 if no pageview data available
                    
                    # Insert into database
                    db.insert_article(
                        wikipedia_id=article_id,
                        title=title,
                        word_count=word_count,
                        page_views=page_views,
                        offset_start=offset_start,
                        offset_end=offset_end,
                        infobox_type=infobox_type
                    )
                    
                    processed_count += 1
                    
                    if processed_count % 500000 == 0:
                        print(f"  Processed {processed_count} articles...")
                        
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: JSON decode error: {e}")
                except Exception as e:
                    errors.append(f"Line {line_num}: {e}")
        
        print(f"✅ Successfully processed {processed_count} articles")
        
        if errors:
            print(f"⚠️  {len(errors)} errors occurred:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"   {error}")
            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more errors")
        
        # Show statistics after import
        stats = db.get_stats()
        print(f"\n📊 Final database statistics:")
        print(f"   Total articles: {stats['total_articles']}")
        print(f"   Top infobox types: {stats['top_infobox_types'][:3]}")
        
    except Exception as e:
        print(f"❌ Error processing corpus file: {e}")
        return

def test_database_queries():
    """Test some database queries after import"""
    db = WikipediaDB()
    
    print(f"\n🧪 Testing database queries...")
    
    # Search for articles
    results = db.search_articles("machine", limit=5)
    if results:
        print(f"🔍 Articles matching 'machine':")
        for article in results:
            infobox_info = f" (infobox: {article['infobox_type']})" if article['infobox_type'] else ""
            print(f"   - {article['title']} ({article['page_views']:,} views){infobox_info}")
    
    # Get a specific article
    if results:
        first_article = results[0]
        print(f"\n📄 Article details for '{first_article['title']}':")
        print(f"   - Word count: {first_article['word_count']}")
        print(f"   - Page views: {first_article['page_views']}")
        print(f"   - Infobox type: {first_article['infobox_type']}")
        print(f"   - Text offsets: {first_article['offset_start']}-{first_article['offset_end']}")

if __name__ == "__main__":
    print("🏗️  Wikipedia Database Import Tool")
    print("=" * 50)
    
    # Setup database first
    from setup_database import setup_database
    db_path = "outputs/wikipedia_data.db"
    
    if not os.path.exists(db_path):
        print("📦 Setting up database...")
        setup_database(db_path)
    
    # Import pageview data
    import_article_metadata()
    
    # Test queries
    test_database_queries()
    
    print(f"\n✅ Process completed!")
    print(f"💾 Database: {os.path.abspath(db_path)}")
    print(f"🔗 Access with: sqlite3 {db_path}")
