import os
import json
import glob
from collections import Counter, defaultdict

# Configuration
RESPONSES_DIR = "llama_absa_responses_2k_v1"  # Directory containing the response files

def analyze_aspects(aspects, required_keys, type_keys):
    """Analyze a list of aspects and return statistics."""
    total_aspects = len(aspects)
    complete_aspects = 0
    missing_keys_counter = Counter()
    category_counter = Counter()
    sentiment_counter = Counter()
    aspect_type_counter = Counter()
    mention_type_counter = Counter()
    confidence_distribution = defaultdict(int)
    
    # Analyze each aspect
    for aspect in aspects:
        # Check if all required keys are present and at least one type key
        has_required_keys = all(key in aspect for key in required_keys)
        has_type_key = any(key in aspect for key in type_keys)
        
        if has_required_keys and has_type_key:
            complete_aspects += 1
            
            # Count categories, sentiments, and types
            category_counter[aspect["category"]] += 1
            sentiment_counter[aspect["sentiment"]] += 1
            
            # Get the type value (either from aspect_type or DOLCEType)
            type_value = None
            if "aspect_type" in aspect:
                type_value = aspect["aspect_type"]
            elif "DOLCEType" in aspect:
                type_value = aspect["DOLCEType"]
            
            if type_value:
                aspect_type_counter[type_value] += 1
                
            mention_type_counter[aspect["mention_type"]] += 1
            
            # Track confidence score distribution (rounded to nearest 0.1)
            conf_rounded = round(aspect["confidence"] * 10) / 10
            confidence_distribution[conf_rounded] += 1
        else:
            # Count missing keys
            for key in required_keys:
                if key not in aspect:
                    missing_keys_counter[key] += 1
            
            # Count missing type keys only if both are missing
            if not has_type_key:
                missing_keys_counter["aspect_type/DOLCEType"] += 1
    
    return {
        "total_aspects": total_aspects,
        "complete_aspects": complete_aspects,
        "missing_keys_counter": missing_keys_counter,
        "category_counter": category_counter,
        "sentiment_counter": sentiment_counter,
        "aspect_type_counter": aspect_type_counter,
        "mention_type_counter": mention_type_counter,
        "confidence_distribution": confidence_distribution
    }

def print_statistics(stats, title, complete_aspects):
    """Print statistics for a given set of aspects."""
    print(f"\n=== {title} ===")
    print(f"Total aspects: {stats['total_aspects']}")
    print(f"Aspects with all required keys: {stats['complete_aspects']} ({(stats['complete_aspects'] / stats['total_aspects']) * 100 if stats['total_aspects'] > 0 else 0:.2f}%)")
    
    if stats['missing_keys_counter']:
        print(f"\n=== {title} - MISSING KEYS ===")
        for key, count in stats['missing_keys_counter'].most_common():
            print(f"{key}: {count} times")
    
    print(f"\n=== {title} - CATEGORY DISTRIBUTION ===")
    for category, count in stats['category_counter'].most_common():
        percentage = (count / complete_aspects) * 100 if complete_aspects > 0 else 0
        print(f"{category}: {count} ({percentage:.2f}%)")
    
    print(f"\n=== {title} - SENTIMENT DISTRIBUTION ===")
    for sentiment, count in stats['sentiment_counter'].most_common():
        percentage = (count / complete_aspects) * 100 if complete_aspects > 0 else 0
        print(f"{sentiment}: {count} ({percentage:.2f}%)")
    
    print(f"\n=== {title} - ASPECT TYPE DISTRIBUTION ===")
    for aspect_type, count in stats['aspect_type_counter'].most_common():
        percentage = (count / complete_aspects) * 100 if complete_aspects > 0 else 0
        print(f"{aspect_type}: {count} ({percentage:.2f}%)")
    
    print(f"\n=== {title} - MENTION TYPE DISTRIBUTION ===")
    for mention_type, count in stats['mention_type_counter'].most_common():
        percentage = (count / complete_aspects) * 100 if complete_aspects > 0 else 0
        print(f"{mention_type}: {count} ({percentage:.2f}%)")
    
    print(f"\n=== {title} - CONFIDENCE SCORE DISTRIBUTION ===")
    for conf, count in sorted(stats['confidence_distribution'].items()):
        percentage = (count / complete_aspects) * 100 if complete_aspects > 0 else 0
        print(f"{conf:.1f}: {count} ({percentage:.2f}%)")

def analyze_responses():
    """Analyze the response files in the specified directory."""
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(RESPONSES_DIR, "*.json"))
    total_files = len(json_files)
    
    if total_files == 0:
        print(f"No JSON files found in {RESPONSES_DIR}")
        return
    
    print(f"Found {total_files} JSON files in {RESPONSES_DIR}")
    
    # Initialize counters and statistics
    files_with_parsed_response = 0
    files_with_ground_truth = 0
    
    # Required keys for aspects (with alternatives)
    required_keys = ["aspect", "category", "sentiment", "confidence", "mention_type", "evidence"]
    type_keys = ["aspect_type", "DOLCEType"]  # Either one of these can be present
    
    # Aggregated statistics
    model_aspects_all = []
    ground_truth_aspects_all = []
    
    # Process each JSON file
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if parsed_response exists
            if "parsed_response" in data and "aspects" in data["parsed_response"]:
                files_with_parsed_response += 1
                model_aspects_all.extend(data["parsed_response"]["aspects"])
            
            # Check if ground_truth exists
            if "ground_truth" in data and "aspects" in data["ground_truth"]:
                files_with_ground_truth += 1
                ground_truth_aspects_all.extend(data["ground_truth"]["aspects"])
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Analyze both sets of aspects
    model_stats = analyze_aspects(model_aspects_all, required_keys, type_keys)
    ground_truth_stats = analyze_aspects(ground_truth_aspects_all, required_keys, type_keys)
    
    # Print overall statistics
    print("\n=== OVERALL STATISTICS ===")
    print(f"Total JSON files: {total_files}")
    print(f"Files with parsed_response: {files_with_parsed_response} ({(files_with_parsed_response / total_files) * 100 if total_files > 0 else 0:.2f}%)")
    print(f"Files with ground_truth: {files_with_ground_truth} ({(files_with_ground_truth / total_files) * 100 if total_files > 0 else 0:.2f}%)")
    
    # Print detailed statistics for both sets
    print_statistics(model_stats, "MODEL RESPONSES", model_stats["complete_aspects"])
    print_statistics(ground_truth_stats, "GROUND TRUTH", ground_truth_stats["complete_aspects"])
    
    # Print comparison statistics
    print("\n=== COMPARISON STATISTICS ===")
    print(f"Average aspects per response (model): {model_stats['total_aspects'] / files_with_parsed_response if files_with_parsed_response > 0 else 0:.2f}")
    print(f"Average aspects per response (ground truth): {ground_truth_stats['total_aspects'] / files_with_ground_truth if files_with_ground_truth > 0 else 0:.2f}")
    
    # Compare category distributions
    print("\n=== CATEGORY DISTRIBUTION COMPARISON ===")
    all_categories = set(model_stats['category_counter'].keys()) | set(ground_truth_stats['category_counter'].keys())
    for category in sorted(all_categories):
        model_count = model_stats['category_counter'].get(category, 0)
        gt_count = ground_truth_stats['category_counter'].get(category, 0)
        model_pct = (model_count / model_stats['complete_aspects']) * 100 if model_stats['complete_aspects'] > 0 else 0
        gt_pct = (gt_count / ground_truth_stats['complete_aspects']) * 100 if ground_truth_stats['complete_aspects'] > 0 else 0
        diff = model_pct - gt_pct
        print(f"{category}: Model: {model_count} ({model_pct:.2f}%) | GT: {gt_count} ({gt_pct:.2f}%) | Diff: {diff:.2f}%")

if __name__ == "__main__":
    analyze_responses()