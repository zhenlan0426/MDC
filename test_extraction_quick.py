#!/usr/bin/env python3
"""
Quick test script to validate extraction utilities on a small subset of training data.

This script provides a fast way to test the extraction system on a few files
before running the full comprehensive test.
"""

import os
import csv
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict

# Import the extraction utilities
from data_extraction_utils import (
    get_comprehensive_patterns,
    extract_identifiers_from_text,
    extract_text_from_pdf,
    normalize_identifier,
    PYMUPDF_AVAILABLE
)

def load_sample_ground_truth(labels_file: str, sample_articles: List[str]) -> Dict[str, Dict[str, str]]:
    """Load ground truth for specific sample articles."""
    ground_truth = {}
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            article_id = row['article_id']
            dataset_id = row['dataset_id']
            type_label = row['type']
            
            # Only process our sample articles
            if article_id not in sample_articles:
                continue
                
            # Skip missing entries
            if dataset_id == 'Missing' or type_label == 'Missing':
                continue
            
            if article_id not in ground_truth:
                ground_truth[article_id] = {}
            
            ground_truth[article_id][dataset_id] = type_label
    
    return ground_truth

def test_sample_extraction():
    """Test extraction on a small sample of files."""
    
    if not PYMUPDF_AVAILABLE:
        print("ERROR: PyMuPDF is not available. Cannot process PDFs.")
        return
    
    # Set up paths
    base_dir = Path(__file__).parent
    pdf_directory = base_dir.parent / "Data" / "train" / "PDF"
    labels_file = base_dir.parent / "Data" / "train_labels.csv"
    
    print("Quick Extraction Test")
    print("=" * 50)
    print(f"PDF Directory: {pdf_directory}")
    print(f"Labels File: {labels_file}")
    print()
    
    # Select a few sample articles to test
    sample_files = [
        "10.7717_peerj.13193.pdf",
        "10.3390_molecules17077645.pdf", 
        "10.1590_1806-90882019000500004.pdf",
        "10.3897_bdj.7.e47369.pdf",
        "10.3390_cryst2031058.pdf"
    ]
    
    sample_articles = [f.replace(".pdf", "") for f in sample_files]
    
    # Load ground truth for sample articles
    print("Loading ground truth for sample articles...")
    ground_truth = load_sample_ground_truth(str(labels_file), sample_articles)
    
    print(f"Ground truth loaded for {len(ground_truth)} articles:")
    for article_id, refs in ground_truth.items():
        print(f"  {article_id}: {len(refs)} references")
        for ref_id, ref_type in refs.items():
            print(f"    - {ref_id} ({ref_type})")
    print()
    
    # Get extraction patterns
    patterns = get_comprehensive_patterns()
    print(f"Loaded {len(patterns)} extraction patterns")
    
    # Test each sample file
    extracted_by_article = defaultdict(set)
    
    for filename in sample_files:
        pdf_path = pdf_directory / filename
        article_id = filename.replace(".pdf", "")
        
        if not pdf_path.exists():
            print(f"Skipping {filename} - file not found")
            continue
        
        print(f"\nProcessing: {filename}")
        
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(str(pdf_path), stop_at_references=True)
            print(f"  Extracted text length: {len(text)} characters")
            
            # Extract identifiers
            results = extract_identifiers_from_text(text, patterns, article_id, text_span_len=100)
            print(f"  Found {len(results)} potential identifiers")
            
            # Normalize and collect
            for _, context, pattern_type, raw_identifier in results:
                normalized = normalize_identifier(raw_identifier, pattern_type)
                extracted_by_article[article_id].add(normalized)
                print(f"    {pattern_type}: {raw_identifier} -> {normalized}")
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
    
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    
    # Compare results
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for article_id in sample_articles:
        print(f"\nArticle: {article_id}")
        
        ground_truth_refs = set(ground_truth.get(article_id, {}).keys())
        extracted_refs = extracted_by_article.get(article_id, set())
        
        # True positives (correct matches)
        tp = ground_truth_refs.intersection(extracted_refs)
        # False positives (extracted but not in ground truth)
        fp = extracted_refs - ground_truth_refs
        # False negatives (in ground truth but not extracted)
        fn = ground_truth_refs - extracted_refs
        
        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)
        
        print(f"  Ground Truth ({len(ground_truth_refs)}): {ground_truth_refs}")
        print(f"  Extracted ({len(extracted_refs)}): {extracted_refs}")
        print(f"  True Positives ({len(tp)}): {tp}")
        print(f"  False Positives ({len(fp)}): {fp}")
        print(f"  False Negatives ({len(fn)}): {fn}")
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nOVERALL METRICS:")
    print(f"  True Positives: {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1_score:.3f}")

if __name__ == "__main__":
    test_sample_extraction() 