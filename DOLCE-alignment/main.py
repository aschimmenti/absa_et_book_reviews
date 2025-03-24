#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import re
import signal
import random
import sys
import time
import pickle
import traceback
from dolce_extractor.core import process_book_aspects, process_books
from dolce_extractor.cache import get_cache, save_cache

# Constants
INPUT_FILE = "book_aspects_filtered.json"
OUTPUT_FILE = "book_aspects_with_dolce.json"
TTL_DIR = "ttl_files"
SAVE_INTERVAL = 10  # Save output every 10 books (changed from 100)
MAX_ASPECTS_PER_CATEGORY = 10  # Maximum number of aspects to process per category
CHECKPOINT_DIR = "checkpoints"  # Directory to store checkpoints

# Global variables for signal handling
processed_books = []
current_interim_filename = ""

def save_processed_books(processed_books, filename=OUTPUT_FILE, interim=False):
    """
    Save processed books to a JSON file.
    
    Args:
        processed_books (list): The list of processed books
        filename (str): The filename to save to
        interim (bool): Whether this is an interim save
    """
    # Create the output filename
    output_filename = filename
    if interim:
        output_filename = f"{filename}.interim"
    
    # Save the processed books to a JSON file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(processed_books, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(processed_books)} processed books to {output_filename}")
    
    # Calculate and print statistics
    total_aspects = 0
    aspects_with_superclass = 0
    
    for book in processed_books:
        if "aspects" in book:
            for aspect_key, aspect_values in book["aspects"].items():
                if isinstance(aspect_values, list):
                    total_aspects += len(aspect_values)
                    for aspect in aspect_values:
                        if isinstance(aspect, dict) and "superclass" in aspect:
                            aspects_with_superclass += 1
    
    if total_aspects > 0:
        percentage = (aspects_with_superclass / total_aspects) * 100
        print(f"Aspects with superclass: {aspects_with_superclass}/{total_aspects} ({percentage:.2f}%)")

def save_checkpoint(index, cache):
    """
    Save a checkpoint with the current processing state.
    
    Args:
        index (int): The current book index
        cache (dict): The current cache state
    """
    # Ensure the checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Create the checkpoint filename with timestamp
    timestamp = int(time.time())
    checkpoint_filename = os.path.join(CHECKPOINT_DIR, f"checkpoint_{timestamp}.pkl")
    
    # Save the checkpoint
    checkpoint_data = {
        "index": index,
        "cache": cache,
        "timestamp": timestamp
    }
    
    with open(checkpoint_filename, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"Saved checkpoint at index {index} to {checkpoint_filename}")

def load_latest_checkpoint():
    """
    Load the latest checkpoint if it exists.
    
    Returns:
        tuple: (index, cache) - The index of the last processed book and the cache
    """
    # Ensure the checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Find all checkpoint files
    checkpoint_files = []
    for filename in os.listdir(CHECKPOINT_DIR):
        if filename.startswith("checkpoint_") and filename.endswith(".pkl"):
            checkpoint_files.append(os.path.join(CHECKPOINT_DIR, filename))
    
    if not checkpoint_files:
        print("No checkpoint found")
        return 0, {}
    
    # Find the latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Found latest checkpoint: {latest_checkpoint}")
    
    # Load the checkpoint
    try:
        with open(latest_checkpoint, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        index = checkpoint_data["index"]
        cache = checkpoint_data["cache"]
        timestamp = checkpoint_data.get("timestamp", "unknown")
        
        print(f"Loaded checkpoint from {latest_checkpoint}")
        print(f"Resuming from index {index} (timestamp: {timestamp})")
        
        return index, cache
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return 0, {}

def load_processed_books(filename):
    """
    Load processed books from a JSON file if it exists.
    
    Args:
        filename (str): The filename to load from
        
    Returns:
        tuple: (processed_books, processed_count) - The list of processed books and the count of processed books
    """
    # Check if the output file exists
    if os.path.exists(filename):
        try:
            # Load the processed books
            with open(filename, 'r', encoding='utf-8') as f:
                processed_books = json.load(f)
            
            print(f"Loaded {len(processed_books)} processed books from {filename}")
            
            # Calculate and print statistics
            total_aspects = 0
            aspects_with_superclass = 0
            
            for book in processed_books:
                if "aspects" in book:
                    for aspect_key, aspect_values in book["aspects"].items():
                        if isinstance(aspect_values, list):
                            total_aspects += len(aspect_values)
                            for aspect in aspect_values:
                                if isinstance(aspect, dict) and "superclass" in aspect:
                                    aspects_with_superclass += 1
            
            if total_aspects > 0:
                percentage = (aspects_with_superclass / total_aspects) * 100
                print(f"Aspects with superclass: {aspects_with_superclass}/{total_aspects} ({percentage:.2f}%)")
            
            return processed_books, len(processed_books)
        except Exception as e:
            print(f"Error loading processed books: {str(e)}")
    
    return [], 0

def signal_handler(sig, frame):
    """
    Handle CTRL+C signal by saving current progress and exiting gracefully.
    """
    print("\nInterrupt received, saving current progress...")
    save_processed_books(processed_books, interim=True)
    print("Exiting...")
    sys.exit(0)

def main():
    """
    Main function to run the script.
    """
    global processed_books, current_interim_filename
    
    # Register signal handler for CTRL+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Ensure the TTL directory exists
    os.makedirs(TTL_DIR, exist_ok=True)
    
    # Ensure the checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Load the latest checkpoint
    start_index, local_cache = load_latest_checkpoint()
    
    # Load the book aspects
    print(f"Loading book aspects from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            books_data = json.load(f)
    except Exception as e:
        print(f"Error loading book aspects: {str(e)}")
        return
    
    print(f"Loaded {len(books_data)} books")
    
    # Load previously processed books if they exist
    processed_books, processed_count = load_processed_books(OUTPUT_FILE)
    
    # If we have a checkpoint, use it to resume processing
    if start_index > 0:
        print(f"Resuming from checkpoint at index {start_index}")
    else:
        print("Starting from the beginning")
    
    # Process the books
    try:
        # Slice the books data to start from the checkpoint
        remaining_books = books_data[start_index:]
        print(f"Processing {len(remaining_books)} remaining books...")
        
        # Process each book
        for i, book_data in enumerate(remaining_books):
            current_index = start_index + i
            print(f"\n\n=== Processing Book {current_index+1}/{len(books_data)}: {book_data.get('title', 'Unknown')} ===")
            
            try:
                # Process the book
                processed_book = process_book_aspects(book_data, MAX_ASPECTS_PER_CATEGORY, local_cache)
                processed_books.append(processed_book)
                
                # Save progress at regular intervals
                if (current_index + 1) % SAVE_INTERVAL == 0:
                    print(f"\n=== Saving progress at {current_index+1}/{len(books_data)} books ===")
                    save_processed_books(processed_books)
                    save_checkpoint(current_index + 1, local_cache)
            except Exception as e:
                print(f"Error processing book: {str(e)}")
                traceback.print_exc()
                print("Saving checkpoint and continuing with the next book...")
                save_checkpoint(current_index + 1, local_cache)
                continue
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        traceback.print_exc()
    finally:
        # Save the final results
        print("\n=== Processing complete, saving final results ===")
        save_processed_books(processed_books)

if __name__ == "__main__":
    main()
