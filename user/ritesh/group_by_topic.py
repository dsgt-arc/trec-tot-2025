import os
import pandas as pd
from collections import defaultdict
import re

def clean_topic_name(topic_name):
    """Clean topic name to create valid filename"""
    # Remove everything after "- Includes" and keep only the first part
    if "- Includes" in topic_name:
        topic_name = topic_name.split("- Includes")[0].strip()
    
    # Remove newlines and extra spaces
    cleaned = re.sub(r'\s+', ' ', topic_name.strip())
    # Remove special characters that might cause issues in filenames
    cleaned = re.sub(r'[<>:"/\\|?*]', '', cleaned)
    # Replace commas and other problematic characters
    cleaned = cleaned.replace(',', '').replace('&', '').replace(' ', '_')
    # remove more than 1 underscore
    cleaned = re.sub(r'_+', '_', cleaned)
    # make everything lowercase
    cleaned = cleaned.lower()
    return cleaned

def group_csv_files_by_topic(input_dir, output_dir):
    """
    Read all CSV files from input_dir and group entries by topic.
    Create separate CSV files for each topic in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store dataframes for each topic
    topic_dfs = defaultdict(list)
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for i, csv_file in enumerate(sorted(csv_files)):
        print(f"Processing file {i+1}/{len(csv_files)}: {csv_file}")
        
        file_path = os.path.join(input_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            
            # Group by predicted_topic
            for topic, group in df.groupby('predicted_topic'):
                # Select only the required columns: id, title, confidence
                topic_data = group[['id', 'title', 'confidence']].copy()
                topic_dfs[topic].append(topic_data)
                
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    # Create separate CSV files for each topic
    print(f"\nCreating {len(topic_dfs)} topic-specific CSV files...")
    
    for topic, dfs in topic_dfs.items():
        if dfs:  # Check if we have data for this topic
            # Concatenate all dataframes for this topic
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Clean topic name for filename
            clean_topic = clean_topic_name(topic)
            filename = f"{clean_topic}.csv"
            output_path = os.path.join(output_dir, filename)
            
            # Save to CSV
            combined_df.to_csv(output_path, index=False)
            print(f"Created: {filename} with {len(combined_df)} entries")
    
    print(f"\nCompleted! Created {len(topic_dfs)} topic-specific CSV files in {output_dir}")

if __name__ == "__main__":
    input_directory = "/workspace/classified_topics_output"
    output_directory = "/workspace/topic_grouped_csv"
    
    group_csv_files_by_topic(input_directory, output_directory) 