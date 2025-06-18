#!/usr/bin/env python3
"""
Test script to evaluate data extraction utilities against training labels.

This script:
1. Loads training labels from CSV
2. Extracts identifiers from training PDFs using extraction utilities
3. Calculates precision and recall metrics
4. Provides detailed analysis and saves results
"""

import os
import sys
import csv
import json
import time
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
from pathlib import Path

# Import the extraction utilities
from data_extraction_utils import (
    get_comprehensive_patterns,
    extract_data_references_from_pdfs,
    normalize_identifier,
    filter_common_false_positives,
    PYMUPDF_AVAILABLE
)

class ExtractionTester:
    """Class to test and evaluate extraction performance."""
    
    def __init__(self, pdf_directory: str, labels_file: str):
        """
        Initialize the tester.
        
        Args:
            pdf_directory: Directory containing training PDFs
            labels_file: Path to training labels CSV file
        """
        self.pdf_directory = pdf_directory
        self.labels_file = labels_file
        self.ground_truth = {}  # {article_id: {dataset_id: type}}
        self.extracted_results = []  # List of extraction results
        self.patterns = get_comprehensive_patterns()
        
    def load_ground_truth(self) -> None:
        """Load ground truth labels from CSV file."""
        print("Loading ground truth labels...")
        
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                article_id = row['article_id']
                dataset_id = row['dataset_id']
                type_label = row['type']
                
                # Skip missing entries
                if dataset_id == 'Missing' or type_label == 'Missing':
                    continue
                
                if article_id not in self.ground_truth:
                    self.ground_truth[article_id] = {}
                
                self.ground_truth[article_id][dataset_id] = type_label
        
        print(f"Loaded ground truth for {len(self.ground_truth)} articles")
        total_refs = sum(len(refs) for refs in self.ground_truth.values())
        print(f"Total ground truth data references: {total_refs}")
    
    def extract_from_pdfs(self) -> None:
        """Extract identifiers from training PDFs."""
        print("Extracting identifiers from PDFs...")
        
        if not PYMUPDF_AVAILABLE:
            print("ERROR: PyMuPDF is not available. Cannot process PDFs.")
            return
        
        def progress_callback(message: str):
            print(f"  {message}")
        
        # Extract from all PDFs
        self.extracted_results = extract_data_references_from_pdfs(
            self.pdf_directory,
            self.patterns,
            progress_callback,
            text_span_len=100,
            stop_at_references=True
        )
        
        # Filter false positives
        print("Filtering false positives...")
        self.extracted_results = filter_common_false_positives(self.extracted_results)
        
        print(f"Final extraction results: {len(self.extracted_results)} potential identifiers")
    
    def normalize_extracted_identifiers(self) -> Dict[str, Set[str]]:
        """
        Normalize extracted identifiers and group by article.
        
        Returns:
            Dict mapping article_id to set of normalized identifiers
        """
        extracted_by_article = defaultdict(set)
        
        for article_id, context, pattern_type, raw_identifier in self.extracted_results:
            normalized = normalize_identifier(raw_identifier, pattern_type)
            extracted_by_article[article_id].add(normalized)
        
        return dict(extracted_by_article)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Returns:
            Dictionary with metrics
        """
        extracted_by_article = self.normalize_extracted_identifiers()
        
        # Count true positives, false positives, false negatives
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Track matches for detailed analysis
        exact_matches = []
        missed_identifiers = []
        false_positive_identifiers = []
        
        # Check each article in ground truth
        for article_id, ground_truth_refs in self.ground_truth.items():
            extracted_refs = extracted_by_article.get(article_id, set())
            ground_truth_set = set(ground_truth_refs.keys())
            
            # Find exact matches
            matches = extracted_refs.intersection(ground_truth_set)
            true_positives += len(matches)
            exact_matches.extend([(article_id, ref) for ref in matches])
            
            # Find missed identifiers (false negatives)
            missed = ground_truth_set - extracted_refs
            false_negatives += len(missed)
            missed_identifiers.extend([(article_id, ref) for ref in missed])
            
            # Find false positives (extracted but not in ground truth)
            false_pos = extracted_refs - ground_truth_set
            false_positives += len(false_pos)
            false_positive_identifiers.extend([(article_id, ref) for ref in false_pos])
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'exact_matches': exact_matches,
            'missed_identifiers': missed_identifiers,
            'false_positive_identifiers': false_positive_identifiers
        }
        
        return metrics
    
    def analyze_pattern_performance(self) -> Dict[str, Dict]:
        """Analyze performance by pattern type."""
        extracted_by_article = self.normalize_extracted_identifiers()
        
        # Group ground truth by pattern type
        pattern_ground_truth = defaultdict(set)
        pattern_extracted = defaultdict(set)
        
        for article_id, refs in self.ground_truth.items():
            for ref_id in refs.keys():
                # Determine pattern type for ground truth identifier
                pattern_type = self.determine_pattern_type(ref_id)
                pattern_ground_truth[pattern_type].add((article_id, ref_id))
        
        # Group extracted by pattern type
        for article_id, context, pattern_type, raw_identifier in self.extracted_results:
            normalized = normalize_identifier(raw_identifier, pattern_type)
            pattern_extracted[pattern_type].add((article_id, normalized))
        
        # Calculate metrics per pattern
        pattern_metrics = {}
        for pattern_type in set(pattern_ground_truth.keys()) | set(pattern_extracted.keys()):
            gt_set = pattern_ground_truth[pattern_type]
            ext_set = pattern_extracted[pattern_type]
            
            tp = len(gt_set.intersection(ext_set))
            fp = len(ext_set - gt_set)
            fn = len(gt_set - ext_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            pattern_metrics[pattern_type] = {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'ground_truth_count': len(gt_set),
                'extracted_count': len(ext_set)
            }
        
        return pattern_metrics
    
    def determine_pattern_type(self, identifier: str) -> str:
        """Determine the pattern type for a given identifier."""
        # Test against each pattern to classify
        for pattern_name, pattern in self.patterns.items():
            if pattern.search(identifier):
                return pattern_name
        return 'unknown'
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        metrics = self.calculate_metrics()
        pattern_metrics = self.analyze_pattern_performance()
        
        report = []
        report.append("=" * 80)
        report.append("DATA EXTRACTION UTILITIES - TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE:")
        report.append(f"  True Positives:  {metrics['true_positives']}")
        report.append(f"  False Positives: {metrics['false_positives']}")
        report.append(f"  False Negatives: {metrics['false_negatives']}")
        report.append(f"  Precision:       {metrics['precision']:.3f}")
        report.append(f"  Recall:          {metrics['recall']:.3f}")
        report.append(f"  F1 Score:        {metrics['f1_score']:.3f}")
        report.append("")
        
        # Pattern-specific performance
        report.append("PERFORMANCE BY PATTERN TYPE:")
        report.append("-" * 50)
        for pattern_type, pm in sorted(pattern_metrics.items()):
            if pm['ground_truth_count'] > 0 or pm['extracted_count'] > 0:
                report.append(f"{pattern_type:20} | P: {pm['precision']:.3f} | R: {pm['recall']:.3f} | F1: {pm['f1_score']:.3f} | GT: {pm['ground_truth_count']:3d} | Ext: {pm['extracted_count']:3d}")
        report.append("")
        
        # Top missed identifiers
        report.append("TOP 10 MISSED IDENTIFIERS:")
        report.append("-" * 40)
        missed_counter = Counter([ref for _, ref in metrics['missed_identifiers']])
        for ref, count in missed_counter.most_common(10):
            report.append(f"  {ref} (missed {count} times)")
        report.append("")
        
        # Top false positives
        report.append("TOP 10 FALSE POSITIVE IDENTIFIERS:")
        report.append("-" * 40)
        fp_counter = Counter([ref for _, ref in metrics['false_positive_identifiers']])
        for ref, count in fp_counter.most_common(10):
            report.append(f"  {ref} (false positive {count} times)")
        report.append("")
        
        return "\n".join(report)
    
    def save_detailed_results(self, output_file: str) -> None:
        """Save detailed results to JSON file."""
        metrics = self.calculate_metrics()
        pattern_metrics = self.analyze_pattern_performance()
        extracted_by_article = self.normalize_extracted_identifiers()
        
        results = {
            'test_metadata': {
                'pdf_directory': self.pdf_directory,
                'labels_file': self.labels_file,
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_articles_with_gt': len(self.ground_truth),
                'total_gt_references': sum(len(refs) for refs in self.ground_truth.values()),
                'total_extracted_results': len(self.extracted_results),
                'pymupdf_available': PYMUPDF_AVAILABLE
            },
            'overall_metrics': {
                'true_positives': metrics['true_positives'],
                'false_positives': metrics['false_positives'],
                'false_negatives': metrics['false_negatives'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            },
            'pattern_metrics': pattern_metrics,
            'detailed_analysis': {
                'exact_matches': metrics['exact_matches'],
                'missed_identifiers': metrics['missed_identifiers'],
                'false_positive_identifiers': metrics['false_positive_identifiers']
            },
            'extracted_by_article': {k: list(v) for k, v in extracted_by_article.items()}
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to: {output_file}")

def main():
    """Main test execution function."""
    # Set up paths
    base_dir = Path(__file__).parent
    pdf_directory = base_dir.parent / "Data" / "train" / "PDF"
    labels_file = base_dir.parent / "Data" / "train_labels.csv"
    
    # Check if files exist
    if not pdf_directory.exists():
        print(f"ERROR: PDF directory not found: {pdf_directory}")
        return
    
    if not labels_file.exists():
        print(f"ERROR: Labels file not found: {labels_file}")
        return
    
    print("Starting extraction utilities performance test...")
    print(f"PDF Directory: {pdf_directory}")
    print(f"Labels File: {labels_file}")
    print(f"PyMuPDF Available: {PYMUPDF_AVAILABLE}")
    print()
    
    # Initialize tester
    tester = ExtractionTester(str(pdf_directory), str(labels_file))
    
    # Load ground truth
    tester.load_ground_truth()
    
    # Extract from PDFs
    tester.extract_from_pdfs()
    
    # Generate and display report
    report = tester.generate_report()
    print(report)
    
    # Save detailed results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = base_dir / f"test_results_{timestamp}.json"
    tester.save_detailed_results(str(results_file))
    
    # Save report to file
    report_file = base_dir / f"test_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Test report saved to: {report_file}")

if __name__ == "__main__":
    main() 