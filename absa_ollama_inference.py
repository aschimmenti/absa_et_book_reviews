import os
import json
import glob
import re
import concurrent.futures
from collections import defaultdict
from difflib import SequenceMatcher
from openai import OpenAI
import datetime
import sys

# Configuration
RESPONSES_DIR = "llama_absa_responses_2k_v3"  # Directory containing the response files

# No need to set API key - it's loaded from environment automatically

def normalize_text(text):
    """Normalize text by removing punctuation, extra spaces, and converting to lowercase."""
    if not isinstance(text, str):
        return ""
    # Remove punctuation and convert to lowercase
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two text strings using SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    
    # Normalize the texts
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    
    # Calculate similarity ratio
    return SequenceMatcher(None, norm_text1, norm_text2).ratio()

def check_gpt_similarity(gt_aspect, pred_aspect, review_text=""):
    """
    Use GPT to check if two aspects refer to the same entity/concept.
    
    Args:
        gt_aspect: Ground truth aspect
        pred_aspect: Predicted aspect
        review_text: The original book review text for context
        
    Returns:
        Boolean indicating if GPT considers the aspects to be related
    """
    try:
        # Extract aspect text
        gt_text = gt_aspect.get("aspect", "")
        pred_text = pred_aspect.get("aspect", "")
        
        if not gt_text or not pred_text:
            return False
        
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a literary review expert that determines if two aspect strings refer to the same entity or concept in a book review."
                },
                {
                    "role": "user",
                    "content": f"""
                    ORIGINAL BOOK REVIEW:
                    "{review_text}"
                    
                    I need to determine if these two aspect strings refer to the same entity or concept in the book review:
                    String 1: "{gt_text}" (from ground truth)
                    String 2: "{pred_text}" (from model prediction)
                    
                    Do these strings refer to the same aspect of the book? Consider the context of the review.
                    """
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "aspect_similarity",
                    "schema": {
                        "type": "object",
                        "required": ["are_similar", "explanation"],
                        "properties": {
                            "are_similar": {
                                "type": "boolean",
                                "description": "Whether the two strings refer to the same aspect of the book"
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Brief explanation of the similarity or difference"
                            }
                        },
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            temperature=0,
            max_tokens=1024,
            top_p=1
        )
        
        # Parse response
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        print(result['are_similar'], gt_text, pred_text)
        return result["are_similar"]
        
    except Exception as e:
        print(f"Error using GPT for similarity check: {e}")
        # Fall back to text similarity check
        return calculate_text_similarity(gt_aspect.get("aspect", ""), pred_aspect.get("aspect", "")) >= 0.7

# Function to execute a single GPT similarity check task
def gpt_similarity_task(task):
    gt_aspect, pred_aspect, review_text = task
    return check_gpt_similarity(gt_aspect, pred_aspect, review_text)

def aspect_match(gt_aspect, pred_aspect, use_gpt=False, review_text=""):
    """
    Determine if a predicted aspect matches a ground truth aspect.
    A prediction is considered correct only if:
    1. The aspect term (entity) is correctly identified
    2. The associated sentiment matches the ground truth
    
    Args:
        gt_aspect: Ground truth aspect
        pred_aspect: Predicted aspect
        use_gpt: Whether to use GPT for aspect similarity checking
        review_text: The original book review text for context
        
    Returns:
        Dict with match results: {
            "aspect_match": Boolean indicating if aspects match semantically,
            "sentiment_match": Boolean indicating if sentiments match,
            "full_match": Boolean indicating if both aspect and sentiment match
        }
    """
    # Check if the aspects are similar using GPT or text similarity
    if use_gpt:
        aspect_similarity_match = check_gpt_similarity(gt_aspect, pred_aspect, review_text)
    else:
        similarity_threshold = 0.7
        aspect_similarity = calculate_text_similarity(gt_aspect.get("aspect", ""), pred_aspect.get("aspect", ""))
        aspect_similarity_match = aspect_similarity >= similarity_threshold
    
    # Check if sentiments match
    sentiment_match = gt_aspect.get("sentiment", "") == pred_aspect.get("sentiment", "")
    
    # A full match requires both aspect identification and sentiment match
    full_match = aspect_similarity_match and sentiment_match
    
    return {
        "aspect_match": aspect_similarity_match,
        "sentiment_match": sentiment_match,
        "full_match": full_match
    }

def evaluate_file(ground_truth_aspects, predicted_aspects, review_text="", use_gpt=False, similarity_results=None):
    """
    Evaluate a single file's aspects against ground truth.
    
    Args:
        ground_truth_aspects: List of ground truth aspects
        predicted_aspects: List of predicted aspects
        review_text: The original book review text for context
        use_gpt: Whether to use GPT for aspect similarity checking
        similarity_results: Pre-computed similarity results when using parallel processing
        
    Returns:
        Dict with match results and metrics
    """
    # Return empty results if either list is empty
    if not ground_truth_aspects or not predicted_aspects:
        return {
            "matches": [],
            "unmatched_gt": list(range(len(ground_truth_aspects))),
            "unmatched_pred": list(range(len(predicted_aspects)))
        }
    
    # Track matches from ground truth to predictions
    matches = []
    matched_pred_indices = set()
    
    # For each ground truth aspect, find the best matching prediction
    for gt_idx, gt_aspect in enumerate(ground_truth_aspects):
        best_match = None
        best_match_idx = None
        
        # Compare with each prediction that hasn't been matched yet
        for pred_idx, pred_aspect in enumerate(predicted_aspects):
            # Skip if this prediction is already matched
            if pred_idx in matched_pred_indices:
                continue
                
            # Check if aspects match
            if use_gpt:
                # Use GPT with review context
                if similarity_results and (gt_idx, pred_idx) in similarity_results:
                    # Use pre-computed similarity result
                    aspect_similarity_match = similarity_results[(gt_idx, pred_idx)]
                    match_result = {
                        "aspect_match": aspect_similarity_match,
                        "sentiment_match": gt_aspect.get("sentiment", "") == pred_aspect.get("sentiment", ""),
                        "full_match": aspect_similarity_match and gt_aspect.get("sentiment", "") == pred_aspect.get("sentiment", "")
                    }
                else:
                    # Fallback to sequential processing if result not available
                    match_result = aspect_match(gt_aspect, pred_aspect, use_gpt, review_text)
            else:
                # Use string similarity without GPT
                match_result = aspect_match(gt_aspect, pred_aspect, use_gpt)
            
            # If we found a full match, use it
            if match_result["full_match"]:
                best_match = match_result
                best_match_idx = pred_idx
                break
                
            # If we found an aspect match but not sentiment, keep it as a potential match
            elif match_result["aspect_match"] and not best_match:
                best_match = match_result
                best_match_idx = pred_idx
        
        # If we found a match, record it
        if best_match:
            matches.append({
                "gt_idx": gt_idx,
                "pred_idx": best_match_idx,
                "match_result": best_match
            })
            matched_pred_indices.add(best_match_idx)
    
    # Identify unmatched ground truth and prediction indices
    matched_gt_indices = {match["gt_idx"] for match in matches}
    unmatched_gt = [i for i in range(len(ground_truth_aspects)) if i not in matched_gt_indices]
    unmatched_pred = [i for i in range(len(predicted_aspects)) if i not in matched_pred_indices]
    
    return {
        "matches": matches,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred
    }

def analyze_responses(use_gpt=False, max_workers=10, output_file=None):
    """
    Analyze the response files and evaluate using the proposed algorithm.
    
    Args:
        use_gpt: Whether to use GPT for aspect matching
        max_workers: Maximum number of parallel workers for GPT calls
        output_file: Path to save the output to a file, if None output is only printed to console
    """
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(RESPONSES_DIR, "*.json"))
    total_files = len(json_files)
    
    if total_files == 0:
        print(f"No JSON files found in {RESPONSES_DIR}")
        return
    
    print(f"Found {total_files} JSON files in {RESPONSES_DIR}")
    print(f"Using GPT for similarity matching: {use_gpt}")
    
    # Initialize counters
    files_with_parsed_response = 0
    files_with_ground_truth = 0
    files_with_both = 0
    
    # For metrics evaluation
    all_file_results = []
    
    # Aggregate dataset-level metrics
    total_gt_aspects = 0
    total_pred_aspects = 0
    total_aspect_matches = 0
    total_full_matches = 0
    
    # Category-level metrics
    category_metrics = defaultdict(lambda: {
        "gt_count": 0,
        "pred_count": 0,
        "aspect_matches": 0,
        "full_matches": 0
    })
    
    # For parallel processing of GPT calls
    if use_gpt:
        # Collect all files and their data first
        file_data_list = []
        
        # Process each JSON file to collect data
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if both parsed_response and ground_truth exist
                has_parsed_response = "parsed_response" in data and "aspects" in data["parsed_response"]
                has_ground_truth = "ground_truth" in data and "aspects" in data["ground_truth"]
                
                # Extract review text for context
                review_text = data.get("user_prompt", "")
                
                if has_parsed_response:
                    files_with_parsed_response += 1
                    model_aspects = data["parsed_response"]["aspects"]
                else:
                    model_aspects = []
                
                if has_ground_truth:
                    files_with_ground_truth += 1
                    ground_truth_aspects = data["ground_truth"]["aspects"]
                else:
                    ground_truth_aspects = []
                
                # Add to list if both ground truth and predictions exist
                if has_parsed_response and has_ground_truth:
                    files_with_both += 1
                    file_data_list.append({
                        "file_path": file_path,
                        "sample_id": data.get("sample_id", os.path.basename(file_path)),
                        "ground_truth_aspects": ground_truth_aspects,
                        "model_aspects": model_aspects,
                        "review_text": review_text
                    })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Collect all GPT similarity check tasks
        all_tasks = []
        task_mapping = {}  # Maps task index to (file_idx, gt_idx, pred_idx)
        
        for file_idx, file_data in enumerate(file_data_list):
            ground_truth_aspects = file_data["ground_truth_aspects"]
            model_aspects = file_data["model_aspects"]
            review_text = file_data["review_text"]
            
            for gt_idx, gt_aspect in enumerate(ground_truth_aspects):
                for pred_idx, pred_aspect in enumerate(model_aspects):
                    task = (gt_aspect, pred_aspect, review_text)
                    task_idx = len(all_tasks)
                    all_tasks.append(task)
                    task_mapping[task_idx] = (file_idx, gt_idx, pred_idx)
        
        # Execute GPT similarity check tasks in parallel
        similarity_results_by_file = [dict() for _ in range(len(file_data_list))]
        
        print(f"Executing {len(all_tasks)} GPT similarity checks in parallel with {max_workers} workers...")
        
        # Process tasks in batches to avoid overwhelming the API
        batch_size = 100
        for i in range(0, len(all_tasks), batch_size):
            batch_tasks = all_tasks[i:i+batch_size]
            batch_indices = list(range(i, min(i+batch_size, len(all_tasks))))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(gpt_similarity_task, task): idx for idx, task in zip(batch_indices, batch_tasks)}
                
                for future in concurrent.futures.as_completed(future_to_idx):
                    task_idx = future_to_idx[future]
                    file_idx, gt_idx, pred_idx = task_mapping[task_idx]
                    
                    try:
                        result = future.result()
                        similarity_results_by_file[file_idx][(gt_idx, pred_idx)] = result
                    except Exception as e:
                        print(f"Error in GPT similarity task: {e}")
                        # Fallback to text similarity
                        gt_aspect, pred_aspect, _ = all_tasks[task_idx]
                        similarity = calculate_text_similarity(gt_aspect.get("aspect", ""), pred_aspect.get("aspect", ""))
                        similarity_results_by_file[file_idx][(gt_idx, pred_idx)] = similarity >= 0.7
            
            print(f"Processed batch {i//batch_size + 1}/{(len(all_tasks) + batch_size - 1)//batch_size}")
        
        # Process each file with pre-computed similarity results
        for file_idx, file_data in enumerate(file_data_list):
            file_path = file_data["file_path"]
            sample_id = file_data["sample_id"]
            ground_truth_aspects = file_data["ground_truth_aspects"]
            model_aspects = file_data["model_aspects"]
            review_text = file_data["review_text"]
            similarity_results = similarity_results_by_file[file_idx]
            
            # Evaluate with pre-computed similarity results
            file_results = evaluate_file(ground_truth_aspects, model_aspects, review_text, use_gpt, similarity_results)
            file_results["file_path"] = file_path
            file_results["sample_id"] = sample_id
            all_file_results.append(file_results)
            
            # Update metrics (same as in the original code)
            total_gt_aspects += len(ground_truth_aspects)
            total_pred_aspects += len(model_aspects)
            
            aspect_matches = sum(1 for match in file_results["matches"] if match["match_result"]["aspect_match"])
            full_matches = sum(1 for match in file_results["matches"] if match["match_result"]["full_match"])
            total_aspect_matches += aspect_matches
            total_full_matches += full_matches
            
            # Update category-level metrics
            for gt_idx, gt_aspect in enumerate(ground_truth_aspects):
                category = gt_aspect.get("category", "UNKNOWN")
                category_metrics[category]["gt_count"] += 1
                
                # Check if this ground truth was matched
                for match in file_results["matches"]:
                    if match["gt_idx"] == gt_idx:
                        if match["match_result"]["aspect_match"]:
                            category_metrics[category]["aspect_matches"] += 1
                        if match["match_result"]["full_match"]:
                            category_metrics[category]["full_matches"] += 1
                        break
            
            # Count predicted aspects by category
            for pred_aspect in model_aspects:
                category = pred_aspect.get("category", "UNKNOWN")
                category_metrics[category]["pred_count"] += 1
    else:
        # Original sequential processing for non-GPT mode
        # Process each JSON file
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if both parsed_response and ground_truth exist
                has_parsed_response = "parsed_response" in data and "aspects" in data["parsed_response"]
                has_ground_truth = "ground_truth" in data and "aspects" in data["ground_truth"]
                
                # Extract review text for context
                review_text = data.get("user_prompt", "")
                
                if has_parsed_response:
                    files_with_parsed_response += 1
                    model_aspects = data["parsed_response"]["aspects"]
                else:
                    model_aspects = []
                
                if has_ground_truth:
                    files_with_ground_truth += 1
                    ground_truth_aspects = data["ground_truth"]["aspects"]
                else:
                    ground_truth_aspects = []
                
                # Evaluate this file if both ground truth and predictions exist
                if has_parsed_response and has_ground_truth:
                    files_with_both += 1
                    
                    # Evaluate aspects with review context
                    file_results = evaluate_file(ground_truth_aspects, model_aspects, review_text, use_gpt)
                    file_results["file_path"] = file_path
                    file_results["sample_id"] = data.get("sample_id", os.path.basename(file_path))
                    all_file_results.append(file_results)
                    
                    # Update dataset-level metrics
                    total_gt_aspects += len(ground_truth_aspects)
                    total_pred_aspects += len(model_aspects)
                    
                    # Count aspect and full matches
                    aspect_matches = sum(1 for match in file_results["matches"] if match["match_result"]["aspect_match"])
                    full_matches = sum(1 for match in file_results["matches"] if match["match_result"]["full_match"])
                    total_aspect_matches += aspect_matches
                    total_full_matches += full_matches
                    
                    # Update category-level metrics
                    for gt_idx, gt_aspect in enumerate(ground_truth_aspects):
                        category = gt_aspect.get("category", "UNKNOWN")
                        category_metrics[category]["gt_count"] += 1
                        
                        # Check if this ground truth was matched
                        for match in file_results["matches"]:
                            if match["gt_idx"] == gt_idx:
                                if match["match_result"]["aspect_match"]:
                                    category_metrics[category]["aspect_matches"] += 1
                                if match["match_result"]["full_match"]:
                                    category_metrics[category]["full_matches"] += 1
                                break
                    
                    # Count predicted aspects by category
                    for pred_aspect in model_aspects:
                        category = pred_aspect.get("category", "UNKNOWN")
                        category_metrics[category]["pred_count"] += 1
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Print overall statistics
    print("\n=== OVERALL STATISTICS ===")
    print(f"Total JSON files: {total_files}")
    print(f"Files with parsed_response: {files_with_parsed_response} ({(files_with_parsed_response / total_files) * 100 if total_files > 0 else 0:.2f}%)")
    print(f"Files with ground_truth: {files_with_ground_truth} ({(files_with_ground_truth / total_files) * 100 if total_files > 0 else 0:.2f}%)")
    print(f"Files with both (used for evaluation): {files_with_both} ({(files_with_both / total_files) * 100 if total_files > 0 else 0:.2f}%)")
    
    # Calculate dataset-level metrics
    print("\n=== DATASET-LEVEL METRICS ===")
    print(f"Total Ground Truth Aspects: {total_gt_aspects}")
    print(f"Total Predicted Aspects: {total_pred_aspects}")
    print(f"Total Aspect Matches (entity identification): {total_aspect_matches}")
    print(f"Total Full Matches (entity + sentiment): {total_full_matches}")
    
    # Calculate F1 scores for aspect matching and full matching
    aspect_precision = total_aspect_matches / total_pred_aspects if total_pred_aspects > 0 else 0
    aspect_recall = total_aspect_matches / total_gt_aspects if total_gt_aspects > 0 else 0
    aspect_f1 = 2 * (aspect_precision * aspect_recall) / (aspect_precision + aspect_recall) if (aspect_precision + aspect_recall) > 0 else 0
    
    full_precision = total_full_matches / total_pred_aspects if total_pred_aspects > 0 else 0
    full_recall = total_full_matches / total_gt_aspects if total_gt_aspects > 0 else 0
    full_f1 = 2 * (full_precision * full_recall) / (full_precision + full_recall) if (full_precision + full_recall) > 0 else 0
    
    print(f"\nAspect Matching (Entity Identification Only):")
    print(f"  Precision: {aspect_precision:.4f}")
    print(f"  Recall: {aspect_recall:.4f}")
    print(f"  F1 Score: {aspect_f1:.4f}")
    
    print(f"\nFull Matching (Entity + Sentiment):")
    print(f"  Precision: {full_precision:.4f}")
    print(f"  Recall: {full_recall:.4f}")
    print(f"  F1 Score: {full_f1:.4f}")
    
    # Calculate per-category metrics
    print("\n=== CATEGORY-LEVEL METRICS ===")
    for category, metrics in sorted(category_metrics.items()):
        # Calculate category-specific F1 scores for the file output
        cat_aspect_prec = metrics["aspect_matches"] / metrics["pred_count"] if metrics["pred_count"] > 0 else 0
        cat_aspect_rec = metrics["aspect_matches"] / metrics["gt_count"] if metrics["gt_count"] > 0 else 0
        cat_aspect_f1 = 2 * (cat_aspect_prec * cat_aspect_rec) / (cat_aspect_prec + cat_aspect_rec) if (cat_aspect_prec + cat_aspect_rec) > 0 else 0
        
        cat_full_prec = metrics["full_matches"] / metrics["pred_count"] if metrics["pred_count"] > 0 else 0
        cat_full_rec = metrics["full_matches"] / metrics["gt_count"] if metrics["gt_count"] > 0 else 0
        cat_full_f1 = 2 * (cat_full_prec * cat_full_rec) / (cat_full_prec + cat_full_rec) if (cat_full_prec + cat_full_rec) > 0 else 0
        
        print(f"\n{category}:")
        print(f"  Ground Truth Count: {metrics['gt_count']}")
        print(f"  Prediction Count: {metrics['pred_count']}")
        print(f"  Aspect Matches: {metrics['aspect_matches']}")
        print(f"  Full Matches: {metrics['full_matches']}")
        
        print(f"  Aspect Matching F1: {cat_aspect_f1:.4f} (Precision: {cat_aspect_prec:.4f}, Recall: {cat_aspect_rec:.4f})")
        print(f"  Full Matching F1: {cat_full_f1:.4f} (Precision: {cat_full_prec:.4f}, Recall: {cat_full_rec:.4f})")
    
    # Error analysis
    print("\n=== ERROR ANALYSIS ===")
    errors = {
        "missed_aspects": total_gt_aspects - total_aspect_matches,  # Ground truths with no matching prediction
        "incorrect_aspects": total_pred_aspects - total_aspect_matches,  # Predictions with no matching ground truth
        "sentiment_errors": total_aspect_matches - total_full_matches  # Aspect matches with incorrect sentiment
    }
    
    print(f"Missed Aspects: {errors['missed_aspects']} ({errors['missed_aspects']/total_gt_aspects*100 if total_gt_aspects > 0 else 0:.2f}% of ground truths)")
    print(f"Incorrect Aspects: {errors['incorrect_aspects']} ({errors['incorrect_aspects']/total_pred_aspects*100 if total_pred_aspects > 0 else 0:.2f}% of predictions)")
    print(f"Sentiment Errors: {errors['sentiment_errors']} ({errors['sentiment_errors']/total_aspect_matches*100 if total_aspect_matches > 0 else 0:.2f}% of matched aspects)")

    if output_file:
        with open(output_file, 'w') as f:
            f.write("=== OVERALL STATISTICS ===\n")
            f.write(f"Total JSON files: {total_files}\n")
            f.write(f"Files with parsed_response: {files_with_parsed_response} ({(files_with_parsed_response / total_files) * 100 if total_files > 0 else 0:.2f}%)\n")
            f.write(f"Files with ground_truth: {files_with_ground_truth} ({(files_with_ground_truth / total_files) * 100 if total_files > 0 else 0:.2f}%)\n")
            f.write(f"Files with both (used for evaluation): {files_with_both} ({(files_with_both / total_files) * 100 if total_files > 0 else 0:.2f}%)\n")
            
            f.write("\n=== DATASET-LEVEL METRICS ===\n")
            f.write(f"Total Ground Truth Aspects: {total_gt_aspects}\n")
            f.write(f"Total Predicted Aspects: {total_pred_aspects}\n")
            f.write(f"Total Aspect Matches (entity identification): {total_aspect_matches}\n")
            f.write(f"Total Full Matches (entity + sentiment): {total_full_matches}\n")
            
            f.write(f"\nAspect Matching (Entity Identification Only):\n")
            f.write(f"  Precision: {aspect_precision:.4f}\n")
            f.write(f"  Recall: {aspect_recall:.4f}\n")
            f.write(f"  F1 Score: {aspect_f1:.4f}\n")
            
            f.write(f"\nFull Matching (Entity + Sentiment):\n")
            f.write(f"  Precision: {full_precision:.4f}\n")
            f.write(f"  Recall: {full_recall:.4f}\n")
            f.write(f"  F1 Score: {full_f1:.4f}\n")
            
            f.write("\n=== CATEGORY-LEVEL METRICS ===\n")
            for category, metrics in sorted(category_metrics.items()):
                f.write(f"\n{category}:\n")
                f.write(f"  Ground Truth Count: {metrics['gt_count']}\n")
                f.write(f"  Prediction Count: {metrics['pred_count']}\n")
                f.write(f"  Aspect Matches: {metrics['aspect_matches']}\n")
                f.write(f"  Full Matches: {metrics['full_matches']}\n")
                
                f.write(f"  Aspect Matching F1: {cat_aspect_f1:.4f} (Precision: {cat_aspect_prec:.4f}, Recall: {cat_aspect_rec:.4f})\n")
                f.write(f"  Full Matching F1: {cat_full_f1:.4f} (Precision: {cat_full_prec:.4f}, Recall: {cat_full_rec:.4f})\n")
            
            f.write("\n=== ERROR ANALYSIS ===\n")
            f.write(f"Missed Aspects: {errors['missed_aspects']} ({errors['missed_aspects']/total_gt_aspects*100 if total_gt_aspects > 0 else 0:.2f}% of ground truths)\n")
            f.write(f"Incorrect Aspects: {errors['incorrect_aspects']} ({errors['incorrect_aspects']/total_pred_aspects*100 if total_pred_aspects > 0 else 0:.2f}% of predictions)\n")
            f.write(f"Sentiment Errors: {errors['sentiment_errors']} ({errors['sentiment_errors']/total_aspect_matches*100 if total_aspect_matches > 0 else 0:.2f}% of matched aspects)\n")

    # Save detailed results to a JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_results_file = f"detailed_results_{timestamp}.json"
    with open(detailed_results_file, 'w') as f:
        json.dump(all_file_results, f, indent=4)

if __name__ == "__main__":
    # Set to True to use GPT for similarity checking (requires API key)
    use_gpt = True
    # Set the number of parallel workers for GPT calls
    max_workers = 10
    # Create output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.txt"
    analyze_responses(use_gpt, max_workers, output_file)