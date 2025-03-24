#!/usr/bin/env python3
"""
Script to compute metrics from detailed results JSON file.
Usage: python compute_metrics.py <detailed_results_json_file>
"""

import json
import sys
import os
import glob
from collections import defaultdict
import datetime

def compute_metrics(detailed_results_file):
    """
    Compute metrics from detailed results JSON file.
    
    Args:
        detailed_results_file (str): Path to the detailed results JSON file
        
    Returns:
        dict: Dictionary containing computed metrics
    """
    print(f"Loading detailed results from: {detailed_results_file}")
    
    with open(detailed_results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Initialize counters
    total_gt_aspects = 0
    total_pred_aspects = 0
    total_aspect_matches = 0
    total_full_matches = 0
    total_entity_type_matches = 0
    
    # Initialize category-level metrics
    category_metrics = defaultdict(lambda: {
        "gt_count": 0,
        "pred_count": 0,
        "aspect_matches": 0,
        "full_matches": 0,
        "entity_type_matches": 0
    })
    
    # Initialize error analysis counters
    errors = {
        "missed_aspects": 0,
        "incorrect_aspects": 0,
        "sentiment_errors": 0,
        "entity_type_errors": 0
    }
    
    # We need to load the original response files to get the DOLCEType information
    response_files = {}
    
    # Process each file's results
    for file_result in results:
        file_path = file_result.get("file_path", "")
        sample_id = file_result.get("sample_id", "")
        
        # Load the response file if not already loaded
        if file_path not in response_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    response_files[file_path] = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Warning: Could not load response file {file_path}: {e}")
                # Try to find the file in the v2 directory structure
                v2_path = os.path.join("llama_absa_responses_2k_v2", "llama_absa_responses", os.path.basename(file_path))
                try:
                    with open(v2_path, 'r', encoding='utf-8') as f:
                        response_files[file_path] = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Warning: Could not load response file {v2_path} either: {e}")
                    # Try with different encodings as a fallback
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            response_files[file_path] = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
                        try:
                            with open(v2_path, 'r', encoding='latin-1') as f:
                                response_files[file_path] = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
                            print(f"Error: Could not load file with any encoding: {e}")
                            response_files[file_path] = None
        
        # Count unmatched ground truth aspects
        unmatched_gt_count = len(file_result.get("unmatched_gt", []))
        errors["missed_aspects"] += unmatched_gt_count
        
        # Count unmatched predicted aspects
        unmatched_pred_count = len(file_result.get("unmatched_pred", []))
        errors["incorrect_aspects"] += unmatched_pred_count
        
        # Get ground truth and predicted aspects from the response file
        response_data = response_files.get(file_path)
        
        if response_data:
            gt_aspects = response_data.get("ground_truth", {}).get("aspects", [])
            pred_aspects = response_data.get("parsed_response", {}).get("aspects", [])
            
            # Process matches
            for match in file_result.get("matches", []):
                match_result = match.get("match_result", {})
                gt_idx = match.get("gt_idx")
                pred_idx = match.get("pred_idx")
                
                # Count aspect matches and full matches
                if match_result.get("aspect_match", False):
                    total_aspect_matches += 1
                    
                    # Count sentiment errors
                    if not match_result.get("sentiment_match", False):
                        errors["sentiment_errors"] += 1
                
                # Count full matches (aspect + sentiment)
                if match_result.get("full_match", False):
                    total_full_matches += 1
                
                # Check entity type match if we have access to the aspects
                if gt_idx is not None and pred_idx is not None and 0 <= gt_idx < len(gt_aspects) and 0 <= pred_idx < len(pred_aspects):
                    gt_entity_type = gt_aspects[gt_idx].get("DOLCEType", "")
                    pred_entity_type = pred_aspects[pred_idx].get("DOLCEType", "")
                    
                    # Count entity type matches
                    if gt_entity_type and pred_entity_type and gt_entity_type == pred_entity_type:
                        total_entity_type_matches += 1
                    elif match_result.get("aspect_match", False):
                        # Only count as an error if the aspect matched but entity type didn't
                        errors["entity_type_errors"] += 1
                    
                    # Update category metrics if we have category information
                    if gt_idx < len(gt_aspects):
                        category = gt_aspects[gt_idx].get("category", "")
                        if category:
                            if match_result.get("aspect_match", False):
                                category_metrics[category]["aspect_matches"] += 1
                            if match_result.get("full_match", False):
                                category_metrics[category]["full_matches"] += 1
                            if gt_entity_type and pred_entity_type and gt_entity_type == pred_entity_type:
                                category_metrics[category]["entity_type_matches"] += 1
        
        # Calculate total ground truth and predicted aspects for this file
        file_gt_count = unmatched_gt_count + len(file_result.get("matches", []))
        file_pred_count = unmatched_pred_count + len(file_result.get("matches", []))
        
        total_gt_aspects += file_gt_count
        total_pred_aspects += file_pred_count
        
        # Update category counts if we have the response data
        if response_data and "ground_truth" in response_data:
            for aspect in response_data["ground_truth"].get("aspects", []):
                category = aspect.get("category", "")
                if category:
                    category_metrics[category]["gt_count"] += 1
            
            for aspect in response_data.get("parsed_response", {}).get("aspects", []):
                category = aspect.get("category", "")
                if category:
                    category_metrics[category]["pred_count"] += 1
    
    # Calculate precision, recall, and F1 scores
    aspect_precision = total_aspect_matches / total_pred_aspects if total_pred_aspects > 0 else 0
    aspect_recall = total_aspect_matches / total_gt_aspects if total_gt_aspects > 0 else 0
    aspect_f1 = 2 * (aspect_precision * aspect_recall) / (aspect_precision + aspect_recall) if (aspect_precision + aspect_recall) > 0 else 0
    
    full_precision = total_full_matches / total_pred_aspects if total_pred_aspects > 0 else 0
    full_recall = total_full_matches / total_gt_aspects if total_gt_aspects > 0 else 0
    full_f1 = 2 * (full_precision * full_recall) / (full_precision + full_recall) if (full_precision + full_recall) > 0 else 0
    
    # Calculate entity type precision, recall, and F1 scores
    entity_type_precision = total_entity_type_matches / total_pred_aspects if total_pred_aspects > 0 else 0
    entity_type_recall = total_entity_type_matches / total_gt_aspects if total_gt_aspects > 0 else 0
    entity_type_f1 = 2 * (entity_type_precision * entity_type_recall) / (entity_type_precision + entity_type_recall) if (entity_type_precision + entity_type_recall) > 0 else 0
    
    # Calculate entity type alignment on matched aspects
    entity_type_alignment_precision = total_entity_type_matches / total_aspect_matches if total_aspect_matches > 0 else 0
    entity_type_alignment_recall = total_entity_type_matches / total_aspect_matches if total_aspect_matches > 0 else 0
    entity_type_alignment_f1 = 2 * (entity_type_alignment_precision * entity_type_alignment_recall) / (entity_type_alignment_precision + entity_type_alignment_recall) if (entity_type_alignment_precision + entity_type_alignment_recall) > 0 else 0
    
    # Compile metrics
    metrics = {
        "total_gt_aspects": total_gt_aspects,
        "total_pred_aspects": total_pred_aspects,
        "total_aspect_matches": total_aspect_matches,
        "total_full_matches": total_full_matches,
        "total_entity_type_matches": total_entity_type_matches,
        "aspect_precision": aspect_precision,
        "aspect_recall": aspect_recall,
        "aspect_f1": aspect_f1,
        "full_precision": full_precision,
        "full_recall": full_recall,
        "full_f1": full_f1,
        "entity_type_precision": entity_type_precision,
        "entity_type_recall": entity_type_recall,
        "entity_type_f1": entity_type_f1,
        "entity_type_alignment_precision": entity_type_alignment_precision,
        "entity_type_alignment_recall": entity_type_alignment_recall,
        "entity_type_alignment_f1": entity_type_alignment_f1,
        "category_metrics": category_metrics,
        "errors": errors
    }
    
    return metrics

def save_metrics(metrics, output_prefix=None):
    """
    Save metrics to text and JSON files.
    
    Args:
        metrics (dict): Dictionary containing computed metrics
        output_prefix (str, optional): Prefix for output files. If None, a timestamp will be used.
    """
    if output_prefix is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"recomputed_metrics_{timestamp}"
    
    # Save metrics to text file
    text_file = f"{output_prefix}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("=== ASPECT-BASED SENTIMENT ANALYSIS EVALUATION ===\n\n")
        
        f.write("=== OVERALL STATISTICS ===\n")
        f.write(f"Total Ground Truth Aspects: {metrics['total_gt_aspects']}\n")
        f.write(f"Total Predicted Aspects: {metrics['total_pred_aspects']}\n")
        f.write(f"Total Aspect Matches (entity identification): {metrics['total_aspect_matches']}\n")
        f.write(f"Total Full Matches (entity + sentiment): {metrics['total_full_matches']}\n")
        f.write(f"Total Entity Type Matches: {metrics['total_entity_type_matches']}\n")
        
        f.write(f"\nAspect Matching (Entity Identification Only):\n")
        f.write(f"  Precision: {metrics['aspect_precision']:.4f}\n")
        f.write(f"  Recall: {metrics['aspect_recall']:.4f}\n")
        f.write(f"  F1 Score: {metrics['aspect_f1']:.4f}\n")
        
        f.write(f"\nFull Matching (Entity + Sentiment):\n")
        f.write(f"  Precision: {metrics['full_precision']:.4f}\n")
        f.write(f"  Recall: {metrics['full_recall']:.4f}\n")
        f.write(f"  F1 Score: {metrics['full_f1']:.4f}\n")
        
        f.write(f"\nEntity Type Matching (Overall):\n")
        f.write(f"  Precision: {metrics['entity_type_precision']:.4f}\n")
        f.write(f"  Recall: {metrics['entity_type_recall']:.4f}\n")
        f.write(f"  F1 Score: {metrics['entity_type_f1']:.4f}\n")
        
        f.write(f"\nEntity Type Alignment (On Matched Aspects):\n")
        f.write(f"  Precision: {metrics['entity_type_alignment_precision']:.4f}\n")
        f.write(f"  Recall: {metrics['entity_type_alignment_recall']:.4f}\n")
        f.write(f"  F1 Score: {metrics['entity_type_alignment_f1']:.4f}\n")
        
        f.write("\n=== CATEGORY-LEVEL METRICS ===\n")
        for category, metrics_data in sorted(metrics["category_metrics"].items()):
            # Calculate category-specific F1 scores
            cat_aspect_prec = metrics_data["aspect_matches"] / metrics_data["pred_count"] if metrics_data["pred_count"] > 0 else 0
            cat_aspect_rec = metrics_data["aspect_matches"] / metrics_data["gt_count"] if metrics_data["gt_count"] > 0 else 0
            cat_aspect_f1 = 2 * (cat_aspect_prec * cat_aspect_rec) / (cat_aspect_prec + cat_aspect_rec) if (cat_aspect_prec + cat_aspect_rec) > 0 else 0
            
            cat_full_prec = metrics_data["full_matches"] / metrics_data["pred_count"] if metrics_data["pred_count"] > 0 else 0
            cat_full_rec = metrics_data["full_matches"] / metrics_data["gt_count"] if metrics_data["gt_count"] > 0 else 0
            cat_full_f1 = 2 * (cat_full_prec * cat_full_rec) / (cat_full_prec + cat_full_rec) if (cat_full_prec + cat_full_rec) > 0 else 0
            
            cat_entity_type_prec = metrics_data["entity_type_matches"] / metrics_data["pred_count"] if metrics_data["pred_count"] > 0 else 0
            cat_entity_type_rec = metrics_data["entity_type_matches"] / metrics_data["gt_count"] if metrics_data["gt_count"] > 0 else 0
            cat_entity_type_f1 = 2 * (cat_entity_type_prec * cat_entity_type_rec) / (cat_entity_type_prec + cat_entity_type_rec) if (cat_entity_type_prec + cat_entity_type_rec) > 0 else 0
            
            f.write(f"\n{category}:\n")
            f.write(f"  Ground Truth Count: {metrics_data['gt_count']}\n")
            f.write(f"  Prediction Count: {metrics_data['pred_count']}\n")
            f.write(f"  Aspect Matches: {metrics_data['aspect_matches']}\n")
            f.write(f"  Full Matches: {metrics_data['full_matches']}\n")
            f.write(f"  Entity Type Matches: {metrics_data['entity_type_matches']}\n")
            
            f.write(f"  Aspect Matching F1: {cat_aspect_f1:.4f} (Precision: {cat_aspect_prec:.4f}, Recall: {cat_aspect_rec:.4f})\n")
            f.write(f"  Full Matching F1: {cat_full_f1:.4f} (Precision: {cat_full_prec:.4f}, Recall: {cat_full_rec:.4f})\n")
            f.write(f"  Entity Type Matching F1: {cat_entity_type_f1:.4f} (Precision: {cat_entity_type_prec:.4f}, Recall: {cat_entity_type_rec:.4f})\n")
        
        f.write("\n=== ERROR ANALYSIS ===\n")
        f.write(f"Missed Aspects: {metrics['errors']['missed_aspects']} ({metrics['errors']['missed_aspects']/metrics['total_gt_aspects']*100 if metrics['total_gt_aspects'] > 0 else 0:.2f}% of ground truths)\n")
        f.write(f"Incorrect Aspects: {metrics['errors']['incorrect_aspects']} ({metrics['errors']['incorrect_aspects']/metrics['total_pred_aspects']*100 if metrics['total_pred_aspects'] > 0 else 0:.2f}% of predictions)\n")
        f.write(f"Sentiment Errors: {metrics['errors']['sentiment_errors']} ({metrics['errors']['sentiment_errors']/metrics['total_aspect_matches']*100 if metrics['total_aspect_matches'] > 0 else 0:.2f}% of matched aspects)\n")
        f.write(f"Entity Type Errors: {metrics['errors']['entity_type_errors']} ({metrics['errors']['entity_type_errors']/metrics['total_aspect_matches']*100 if metrics['total_aspect_matches'] > 0 else 0:.2f}% of matched aspects)\n")
    
    # Save metrics to JSON file
    json_file = f"{output_prefix}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {text_file} and {json_file}")
    return text_file, json_file

def print_metrics(metrics):
    """
    Print metrics to console.
    
    Args:
        metrics (dict): Dictionary containing computed metrics
    """
    print("\n=== ASPECT-BASED SENTIMENT ANALYSIS EVALUATION ===\n")
    
    print("=== OVERALL STATISTICS ===")
    print(f"Total Ground Truth Aspects: {metrics['total_gt_aspects']}")
    print(f"Total Predicted Aspects: {metrics['total_pred_aspects']}")
    print(f"Total Aspect Matches (entity identification): {metrics['total_aspect_matches']}")
    print(f"Total Full Matches (entity + sentiment): {metrics['total_full_matches']}")
    print(f"Total Entity Type Matches: {metrics['total_entity_type_matches']}")
    
    print(f"\nAspect Matching (Entity Identification Only):")
    print(f"  Precision: {metrics['aspect_precision']:.4f}")
    print(f"  Recall: {metrics['aspect_recall']:.4f}")
    print(f"  F1 Score: {metrics['aspect_f1']:.4f}")
    
    print(f"\nFull Matching (Entity + Sentiment):")
    print(f"  Precision: {metrics['full_precision']:.4f}")
    print(f"  Recall: {metrics['full_recall']:.4f}")
    print(f"  F1 Score: {metrics['full_f1']:.4f}")
    
    print(f"\nEntity Type Matching (Overall):")
    print(f"  Precision: {metrics['entity_type_precision']:.4f}")
    print(f"  Recall: {metrics['entity_type_recall']:.4f}")
    print(f"  F1 Score: {metrics['entity_type_f1']:.4f}")
    
    print(f"\nEntity Type Alignment (On Matched Aspects):")
    print(f"  Precision: {metrics['entity_type_alignment_precision']:.4f}")
    print(f"  Recall: {metrics['entity_type_alignment_recall']:.4f}")
    print(f"  F1 Score: {metrics['entity_type_alignment_f1']:.4f}")
    
    print("\n=== ERROR ANALYSIS ===")
    print(f"Missed Aspects: {metrics['errors']['missed_aspects']} ({metrics['errors']['missed_aspects']/metrics['total_gt_aspects']*100 if metrics['total_gt_aspects'] > 0 else 0:.2f}% of ground truths)")
    print(f"Incorrect Aspects: {metrics['errors']['incorrect_aspects']} ({metrics['errors']['incorrect_aspects']/metrics['total_pred_aspects']*100 if metrics['total_pred_aspects'] > 0 else 0:.2f}% of predictions)")
    print(f"Sentiment Errors: {metrics['errors']['sentiment_errors']} ({metrics['errors']['sentiment_errors']/metrics['total_aspect_matches']*100 if metrics['total_aspect_matches'] > 0 else 0:.2f}% of matched aspects)")
    print(f"Entity Type Errors: {metrics['errors']['entity_type_errors']} ({metrics['errors']['entity_type_errors']/metrics['total_aspect_matches']*100 if metrics['total_aspect_matches'] > 0 else 0:.2f}% of matched aspects)")

def main():
    """Main function to run the script."""
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <detailed_results_json_file> [output_prefix]")
        sys.exit(1)
    
    detailed_results_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else None
    
    metrics = compute_metrics(detailed_results_file)
    print_metrics(metrics)
    save_metrics(metrics, output_prefix)

if __name__ == "__main__":
    main()
