#!/usr/bin/env python3
"""
Run comprehensive test on the full training dataset.
"""

import time
from test_extraction_improved import ImprovedExtractionTester

def main():
    """Run the comprehensive test."""
    # Set up paths
    from pathlib import Path
    base_dir = Path(__file__).parent
    pdf_directory = base_dir.parent / "Data" / "train" / "PDF"
    labels_file = base_dir.parent / "Data" / "train_labels.csv"
    
    print("=" * 80)
    print("RUNNING COMPREHENSIVE TEST ON FULL DATASET")
    print("=" * 80)
    print(f"PDF Directory: {pdf_directory}")
    print(f"Labels File: {labels_file}")
    print()
    
    # Initialize tester
    tester = ImprovedExtractionTester(str(pdf_directory), str(labels_file))
    
    # Load ground truth
    tester.load_ground_truth()
    
    # Run extraction on full dataset
    print("\nRunning extraction on ALL training files...")
    start_time = time.time()
    tester.extract_from_pdfs()  # No max_files limit = full dataset
    end_time = time.time()
    
    print(f"Extraction completed in {end_time - start_time:.1f} seconds")
    
    # Generate comprehensive report
    report = tester.generate_comprehensive_report()
    print(report)
    
    # Save results with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_prefix = f"full_dataset_test_{timestamp}"
    tester.save_results(output_prefix)
    
    print("\n" + "=" * 80)
    print("FULL DATASET TEST COMPLETE")
    print("=" * 80)
    print(f"Results saved with prefix: {output_prefix}")

if __name__ == "__main__":
    main() 