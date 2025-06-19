#!/usr/bin/env python3
"""
Improved test script to evaluate data extraction utilities against training labels.

This version includes:
- Better filtering of false positives
- Analysis of actual identifier types in ground truth
- More focused pattern matching
- Detailed performance metrics
"""

import os
import csv
import json
import time
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
from pathlib import Path

# Import the extraction utilities
from data_extraction_utils import (
    get_comprehensive_patterns,
    extract_data_references_from_pdfs,
    filter_common_false_positives,
    PYMUPDF_AVAILABLE
)

class ImprovedExtractionTester:
    """Improved class to test and evaluate extraction performance."""
    
    def __init__(self, pdf_directory: str, labels_file: str):
        """Initialize the tester."""
        self.pdf_directory = pdf_directory
        self.labels_file = labels_file
        self.ground_truth = {}  # {article_id: {dataset_id: type}}
        self.extracted_results = []
        self.patterns = get_comprehensive_patterns()
        
        # Analyze ground truth patterns
        self.gt_pattern_analysis = {}
        
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
        
        # Analyze patterns in ground truth
        self.analyze_ground_truth_patterns()
    
    def analyze_ground_truth_patterns(self) -> None:
        """Analyze what types of identifiers are in the ground truth."""
        print("\nAnalyzing ground truth identifier patterns...")
        
        pattern_counts = Counter()
        all_identifiers = []
        
        for article_id, refs in self.ground_truth.items():
            for ref_id in refs.keys():
                all_identifiers.append(ref_id)
                pattern_type = self.determine_pattern_type(ref_id)
                pattern_counts[pattern_type] += 1
        
        print("Ground truth identifier types:")
        for pattern_type, count in pattern_counts.most_common():
            print(f"  {pattern_type}: {count}")
        
        # Sample examples for each pattern type
        pattern_examples = defaultdict(list)
        for ref_id in all_identifiers:
            pattern_type = self.determine_pattern_type(ref_id)
            if len(pattern_examples[pattern_type]) < 3:
                pattern_examples[pattern_type].append(ref_id)
        
        print("\nExample identifiers by type:")
        for pattern_type, examples in pattern_examples.items():
            print(f"  {pattern_type}: {examples}")
        
        self.gt_pattern_analysis = {
            'pattern_counts': dict(pattern_counts),
            'pattern_examples': dict(pattern_examples),
            'total_identifiers': len(all_identifiers)
        }
    
    def extract_from_pdfs(self, max_files: int = None) -> None:
        """Extract identifiers from training PDFs."""
        print("Extracting identifiers from PDFs...")
        
        if not PYMUPDF_AVAILABLE:
            print("ERROR: PyMuPDF is not available. Cannot process PDFs.")
            return
        
        # For testing, limit to subset if specified
        if max_files:
            print(f"  Running on subset of {max_files} files for testing...")
            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
            pdf_files = pdf_files[:max_files]
            
            # Create temporary directory with subset
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp()
            for pdf_file in pdf_files:
                shutil.copy(
                    os.path.join(self.pdf_directory, pdf_file),
                    os.path.join(temp_dir, pdf_file)
                )
            pdf_dir_to_use = temp_dir
        else:
            pdf_dir_to_use = self.pdf_directory
        
        # Extract from PDFs (filtering is now integrated into the main function)
        self.extracted_results = extract_data_references_from_pdfs(
            pdf_dir_to_use,
            self.patterns,
            text_span_len=100,
            stop_at_references=True,
            apply_false_positive_filtering=False
        )
        
        # Clean up temporary directory if used
        if max_files:
            shutil.rmtree(temp_dir)
        
        print(f"Final extraction results: {len(self.extracted_results)} potential identifiers")
    
    def determine_pattern_type(self, identifier: str) -> str:
        """Determine the pattern type for a given identifier."""
        # Test against each pattern to classify
        for pattern_name, pattern in self.patterns.items():
            if pattern.search(identifier):
                return pattern_name
        return 'unknown'
    
    def calculate_detailed_metrics(self) -> Dict:
        """Calculate detailed precision, recall, and F1 metrics."""
        extracted_by_article = defaultdict(set)
        
        for article_id, context, pattern_type, normalized_identifier in self.extracted_results:
            extracted_by_article[article_id].add(normalized_identifier)
        
        # Calculate metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
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
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'exact_matches': exact_matches,
            'missed_identifiers': missed_identifiers,
            'false_positive_identifiers': false_positive_identifiers,
            'extracted_by_article': dict(extracted_by_article)
        }
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive test report."""
        metrics = self.calculate_detailed_metrics()
        
        report = []
        report.append("=" * 80)
        report.append("IMPROVED DATA EXTRACTION UTILITIES - TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Test metadata
        report.append("TEST METADATA:")
        report.append(f"  PDF Directory: {self.pdf_directory}")
        report.append(f"  Labels File: {self.labels_file}")
        report.append(f"  Test Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"  PyMuPDF Available: {PYMUPDF_AVAILABLE}")
        report.append("")
        
        # Ground truth analysis
        report.append("GROUND TRUTH ANALYSIS:")
        report.append(f"  Total articles with data refs: {len(self.ground_truth)}")
        report.append(f"  Total data references: {self.gt_pattern_analysis['total_identifiers']}")
        report.append("")
        
        report.append("  Identifier types in ground truth:")
        for pattern_type, count in self.gt_pattern_analysis['pattern_counts'].items():
            pct = 100 * count / self.gt_pattern_analysis['total_identifiers']
            report.append(f"    {pattern_type:20}: {count:4d} ({pct:5.1f}%)")
        report.append("")
        
        # Overall performance
        report.append("OVERALL PERFORMANCE:")
        report.append(f"  True Positives:  {metrics['true_positives']}")
        report.append(f"  False Positives: {metrics['false_positives']}")
        report.append(f"  False Negatives: {metrics['false_negatives']}")
        report.append(f"  Precision:       {metrics['precision']:.3f}")
        report.append(f"  Recall:          {metrics['recall']:.3f}")
        report.append(f"  F1 Score:        {metrics['f1_score']:.3f}")
        report.append("")
        
        # Top missed identifiers
        if metrics['missed_identifiers']:
            report.append("TOP MISSED IDENTIFIERS (FALSE NEGATIVES):")
            report.append("-" * 50)
            missed_counter = Counter([ref for _, ref in metrics['missed_identifiers']])
            for ref, count in missed_counter.most_common(15):
                pattern_type = self.determine_pattern_type(ref)
                report.append(f"  {ref:50} | {pattern_type:15} | Missed {count:2d} times")
            report.append("")
        
        # Top false positives
        if metrics['false_positive_identifiers']:
            report.append("TOP FALSE POSITIVE IDENTIFIERS:")
            report.append("-" * 50)
            fp_counter = Counter([ref for _, ref in metrics['false_positive_identifiers']])
            for ref, count in fp_counter.most_common(15):
                pattern_type = self.determine_pattern_type(ref)
                report.append(f"  {ref:50} | {pattern_type:15} | FP {count:2d} times")
            report.append("")
        
        # Success stories
        if metrics['exact_matches']:
            report.append("SUCCESSFUL MATCHES (SAMPLE):")
            report.append("-" * 40)
            for i, (article_id, ref) in enumerate(metrics['exact_matches'][:10]):
                pattern_type = self.determine_pattern_type(ref)
                report.append(f"  {ref:40} | {pattern_type:15} | {article_id}")
                if i >= 9:
                    break
            if len(metrics['exact_matches']) > 10:
                report.append(f"  ... and {len(metrics['exact_matches']) - 10} more")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, output_prefix: str) -> None:
        """Save detailed results to files."""
        metrics = self.calculate_detailed_metrics()
        
        # Create results dictionary
        results = {
            'test_metadata': {
                'pdf_directory': self.pdf_directory,
                'labels_file': self.labels_file,
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'pymupdf_available': PYMUPDF_AVAILABLE
            },
            'ground_truth_analysis': self.gt_pattern_analysis,
            'extraction_metrics': {
                'true_positives': metrics['true_positives'],
                'false_positives': metrics['false_positives'],
                'false_negatives': metrics['false_negatives'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            },
            'detailed_results': {
                'exact_matches': metrics['exact_matches'],
                'missed_identifiers': metrics['missed_identifiers'],
                'false_positive_identifiers': metrics['false_positive_identifiers'],
                'extracted_by_article': {k: list(v) for k, v in metrics['extracted_by_article'].items()}
            }
        }
        
        # Save JSON results
        json_file = f"{output_prefix}_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {json_file}")
        
        # Save text report
        report = self.generate_comprehensive_report()
        txt_file = f"{output_prefix}_report.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Test report saved to: {txt_file}")

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
    
    print("Starting improved extraction utilities performance test...")
    print(f"PDF Directory: {pdf_directory}")
    print(f"Labels File: {labels_file}")
    print()
    
    # Initialize tester
    tester = ImprovedExtractionTester(str(pdf_directory), str(labels_file))
    
    # Load ground truth
    tester.load_ground_truth()
    
    # For initial testing, limit to subset
    print("\n" + "="*60)
    print("RUNNING TEST ON SAMPLE OF FILES")
    print("="*60)
    tester.extract_from_pdfs(max_files=20)  # Test on 20 files first
    
    # Generate and display report
    report = tester.generate_comprehensive_report()
    print(report)
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_prefix = f"improved_test_{timestamp}"
    tester.save_results(output_prefix)
    
    # Ask if user wants to run full test
    print("\n" + "="*60)
    print("Sample test complete. To run on all files, uncomment the full test section below.")
    print("="*60)
    
    # Uncomment below to run full test
    # print("RUNNING FULL TEST ON ALL FILES...")
    # tester.extract_from_pdfs()  # Full test
    # full_report = tester.generate_comprehensive_report()
    # print(full_report)
    # tester.save_results(f"full_test_{timestamp}")

if __name__ == "__main__":
    main() 