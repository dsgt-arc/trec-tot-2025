#!/usr/bin/env python3
"""
Test script for Wikipedia sampling functionality
Run individual sampling components for testing and debugging
"""

from run_sampling import WikipediaSampler

def test_filtering_criteria():
    """Test filtering criteria calculation"""
    print("Testing filtering criteria...")
    sampler = WikipediaSampler()
    
    criteria = sampler.get_filtering_criteria()
    
    print(f"\nFiltering criteria results:")
    print(f"   - Min word count: {criteria['min_word_count']}")
    print(f"   - Max word count: {criteria['max_word_count']}")
    print(f"   - Min page views: {criteria['min_page_views']}")
    print(f"   - Total eligible articles: {criteria['total_eligible']}")
    
    return criteria

def test_cutoffs():
    """Test cutoff calculations"""
    print("Testing cutoff calculations...")
    sampler = WikipediaSampler()
    
    print("\nWord count cutoffs:")
    word_cutoffs = sampler.calculate_word_count_cutoffs()
    
    print("\nPage view cutoffs:")
    page_cutoffs = sampler.calculate_page_view_cutoffs()
    
    return word_cutoffs, page_cutoffs

def test_category_counts():
    """Test category counting"""
    print("Testing category counts...")
    sampler = WikipediaSampler()
    categories = sampler.get_category_counts()
    
    print(f"\nTop 20 categories:")
    for i, (category, count) in enumerate(list(categories.items())[:20]):
        sample_size = sampler.determine_sample_size(count)
        print(f"{i+1:2d}. {category}: {count} articles → {sample_size} samples")
    
    return categories

def test_limited_sampling():
    """Test sampling with limited categories"""
    print("Testing limited sampling (first 5 categories)...")
    sampler = WikipediaSampler()
    
    # Sample articles with test mode - limit to 5 categories for quick testing
    articles = sampler.sample_articles_by_category(test_mode=True, test_limit=5)
    
    print(f"\nSampled {len(articles)} articles from limited categories")
    if articles:
        print("\nAll sampled articles:")
        for article in articles:
            print(f"   - {article[0]}: {article[1]} ({article[2]})")
    
    return articles

def test_top_bottom_page_views():
    """Test top and bottom page views functionality"""
    print("Testing top and bottom page views...")
    sampler = WikipediaSampler()
    
    # Get top 20 and bottom 20 pages by page views
    top_bottom_data = sampler.get_top_bottom_page_views(top_n=20, bottom_n=20)
    
    print(f"\nRetrieved {len(top_bottom_data['top'])} top pages and {len(top_bottom_data['bottom'])} bottom pages")
    
    return top_bottom_data

def test_sampling():
    """Test the sampling process"""
    print("Testing sampling process...")
    sampler = WikipediaSampler()
    
    # Sample articles
    articles = sampler.sample_articles_by_category()
    
    print(f"\nSampled {len(articles)} articles")
    if articles:
        print("\nSample of results:")
        for article in articles[:10]:  # Show first 10
            print(f"   - {article[0]}: {article[1]} ({article[2]})")
        
        if len(articles) > 10:
            print(f"   ... and {len(articles) - 10} more")
    
    return articles

def main():
    """Run all tests"""
    print("Running Wikipedia Sampler Tests")
    print("=" * 50)
    
    # Test 1: Cutoffs
    test_cutoffs()
    
    print("\n" + "=" * 50)
    
    # Test 2: Filtering Criteria
    test_filtering_criteria()
    
    print("\n" + "=" * 50)
    
    # Test 3: Categories
    test_category_counts()
    
    print("\n" + "=" * 50)
    
    # Test 4: Top and Bottom Page Views
    test_top_bottom_page_views()
    
    print("\n" + "=" * 50)
    
    # Test 5: Limited Sampling
    test_limited_sampling()
    
    print("\n" + "=" * 50)
    
    # Test 6: Full Sampling (commented out)
    # test_sampling()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    # main()
    test_top_bottom_page_views()
