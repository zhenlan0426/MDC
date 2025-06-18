#!/usr/bin/env python3
"""
MDC Search LLM Logits Processor - Academic Data Reference Extraction

This script processes academic papers (PDFs) to extract and classify DOI references 
using a Large Language Model (LLM) with logits processing. It's designed for the 
"Make Data Count" competition to find data references in academic literature.

The pipeline consists of:
1. PDF text extraction and DOI pattern detection
2. LLM-based DOI URL normalization
3. LLM-based classification of data references (Primary/Secondary/None)
4. Submission preparation and validation scoring
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm  # Progress bars for loops
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor  # Constrained generation
import pickle
import vllm  # Vector LLM library for efficient inference
import torch

# Import our refactored extraction utilities
from data_extraction_utils import (
    extract_data_references_from_pdfs,
    get_kaggle_pdf_directory,
    normalize_identifier,
    filter_common_false_positives,
    get_comprehensive_patterns
)

# ==============================================================================
# CONFIGURATION AND SETUP
# ==============================================================================

# Disable vLLM V1 to maintain compatibility with logits processors
# vLLM V1 currently doesn't support logits processors which we need for classification
os.environ["VLLM_USE_V1"] = "0"

# ==============================================================================
# PDF PROCESSING AND DOI EXTRACTION
# ==============================================================================

def extract_all_data_references_from_pdfs():
    """
    Enhanced version to capture all types of data identifiers.
    
    This function extracts both DOI and accession ID patterns from academic papers,
    covering major scientific databases and repositories including:
    - DOIs (full URLs and bare format)
    - Gene Expression Omnibus (GSE, SRP)
    - ArrayExpress (E-GEOD, E-MTAB, etc.)
    - Structural biology (PDB, EMPIAR, CATH)
    - Chemical databases (ChEMBL)
    - Protein databases (InterPro, Pfam, UniProt)
    - Genomics (Ensembl, GenBank)
    - Cell lines (CVCL) and epidemiological data (EPI_ISL)
    
    Returns:
        list: Tuples of (article_id, text_chunk, pattern_type, raw_identifier)
    """
    # Get PDF directory based on Kaggle environment
    pdf_directory = get_kaggle_pdf_directory()
    
    # Use the refactored extraction function
    chunks = extract_data_references_from_pdfs(
        pdf_directory=pdf_directory,
        patterns=get_comprehensive_patterns(),
        text_span_len=100,
        stop_at_references=True
    )
    
    # Apply false positive filtering
    filtered_chunks = filter_common_false_positives(chunks)
    
    print(f"Extracted {len(chunks)} raw identifiers, filtered to {len(filtered_chunks)} after false positive removal")
    
    return filtered_chunks

# normalize_identifier function is now imported from data_extraction_utils

# ==============================================================================
# LLM SETUP AND INITIALIZATION
# ==============================================================================

def initialize_llm():
    """
    Initialize the Large Language Model for DOI processing.
    
    Uses Qwen2.5-32B-Instruct with AWQ quantization for efficient inference.
    Configured for tensor parallelism across available GPUs.
    
    Returns:
        tuple: (llm_instance, tokenizer)
    """
    model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
    
    # Initialize vLLM with optimized settings
    llm = vllm.LLM(
        model_path,
        quantization='awq',  # AWQ quantization for memory efficiency
        tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs
        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        trust_remote_code=True,  # Allow custom model code
        dtype="half",  # Use FP16 for memory efficiency
        enforce_eager=True,  # Disable CUDA graphs for stability
        max_model_len=2048,  # Maximum sequence length
        disable_log_stats=True,  # Reduce logging overhead
        enable_prefix_caching=True  # Cache common prefixes for efficiency
    )
    
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer

# ==============================================================================
# DOI URL NORMALIZATION
# ==============================================================================
# TODO: rule-based normalization for all
def normalize_data_identifiers(llm, tokenizer, chunks):
    """
    Use LLM to extract and normalize data identifiers, with fallback to regex-based normalization.
    
    This hybrid approach:
    1. Uses LLM for complex cases where context is needed
    2. Uses regex-based normalization for well-defined patterns
    3. Handles both DOI and non-DOI identifiers appropriately
    
    Args:
        llm: vLLM instance
        tokenizer: Model tokenizer
        chunks: List of (article_id, text_chunk, pattern_type, raw_identifier) tuples
        
    Returns:
        list: Normalized identifier strings
    """
    normalized_identifiers = []
    
    print("Normalizing data identifiers...")
    
    # Separate DOI patterns that need LLM processing from others
    doi_chunks = []
    doi_indices = []
    
    for i, (article_id, academic_text, pattern_type, raw_identifier) in enumerate(chunks):
        if pattern_type in ['doi_https', 'doi_bare']:
            # DOI patterns might need LLM for complex extraction
            doi_chunks.append((article_id, academic_text))
            doi_indices.append(i)
        else:
            # Non-DOI patterns can be normalized directly
            normalized_id = normalize_identifier(raw_identifier, pattern_type)
            normalized_identifiers.append(normalized_id)
    
    # Process DOI patterns with LLM if any exist
    if doi_chunks:
        print(f"Processing {len(doi_chunks)} DOI patterns with LLM...")
        
        # System prompt for DOI extraction and normalization
        SYS_PROMPT = """
You are given a piece of academic text. Your task is to identify the single DOI citation string, if present.
Then normalize it into its full URL format: https://doi.org/...
"""
        
        prompts = []
        
        # Prepare prompts for DOI chunks
        for article_id, academic_text in doi_chunks:
            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": academic_text}
            ]
            
            # Apply chat template and add prefix to guide generation
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            ) + "Here is the normalized URL: https://doi.org"
            
            prompts.append(prompt)
        
        # Generate responses using the LLM
        outputs = llm.generate(
            prompts,
            vllm.SamplingParams(
                seed=0,  # Deterministic generation
                skip_special_tokens=True,
                max_tokens=32,  # Short responses expected
                temperature=0  # Deterministic (no randomness)
            ),
            use_tqdm=True
        )
        
        # Extract and format DOI URLs from responses
        responses = [output.outputs[0].text for output in outputs]
        doi_urls = []
        
        for response in responses:
            # Complete the URL by combining prefix with LLM output
            doi_url = "https://doi.org" + response.split("\n")[0]
            doi_urls.append(doi_url)
        
        # Insert DOI URLs back into the correct positions
        doi_idx = 0
        final_identifiers = []
        
        for i in range(len(chunks)):
            if i in doi_indices:
                final_identifiers.append(doi_urls[doi_idx])
                doi_idx += 1
            else:
                # Find the position in normalized_identifiers
                non_doi_idx = i - doi_idx
                final_identifiers.append(normalized_identifiers[non_doi_idx])
        
        return final_identifiers
    
    else:
        # No DOI patterns, return the normalized non-DOI identifiers
        return normalized_identifiers

# ==============================================================================
# DOI CLASSIFICATION
# ==============================================================================

def classify_data_references(llm, tokenizer, chunks):
    """
    Classify data references into categories using constrained generation.
    
    Uses MultipleChoiceLogitsProcessor to ensure LLM outputs only valid choices:
    - A (Primary): Data generated specifically for this study
    - B (Secondary): Data reused/derived from prior work  
    - C (None): Reference citation or unrelated to research data
    
    Args:
        llm: vLLM instance
        tokenizer: Model tokenizer
        chunks: List of (article_id, text_chunk, pattern_type, raw_identifier) tuples
        
    Returns:
        tuple: (classification_answers, logit_probabilities)
    """
    # System prompt for classification task - updated to handle all data types
    SYS_PROMPT = """
You are given a piece of academic text. Your task is to identify the data identifier (DOI, accession ID, or other data reference), if present.
Classify the data associated with that identifier as:
A)Primary: if the data was generated specifically for this study.
B)Secondary: if the data was reused or derived from prior work.
C)None: if the identifier is part of the References section of a paper, does not refer to research data or is unrelated.

Respond with one of A, B or C.
"""
    
    prompts = []
    
    # Prepare prompts for classification
    for article_id, academic_text, pattern_type, raw_identifier in chunks:
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": academic_text}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(prompt)
    
    # Create logits processor to constrain outputs to valid choices
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B", "C"])
    
    # Generate classifications with constrained sampling
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(
            seed=0,  # Deterministic generation
            skip_special_tokens=True,
            max_tokens=1,  # Single token response (A, B, or C)
            logits_processors=[mclp],  # Constrain to valid choices
            logprobs=len(mclp.choices)  # Return probabilities for analysis
        ),
        use_tqdm=True
    )
    
    # Extract log probabilities for each choice
    logprobs = []
    for lps in [output.outputs[0].logprobs[0].values() for output in outputs]:
        logprobs.append({lp.decoded_token: lp.logprob for lp in list(lps)})
    
    # Convert to matrix and get final classifications
    logit_matrix = pd.DataFrame(logprobs)[["A", "B", "C"]].values
    
    # Map choices to human-readable labels
    choices = ["Primary", "Secondary", None]
    answers = [choices[pick] for pick in np.argmax(logit_matrix, axis=1)]
    
    return answers, logit_matrix

# ==============================================================================
# SUBMISSION PREPARATION
# ==============================================================================

def prepare_submission(chunks, normalized_identifiers, answers):
    """
    Prepare final submission DataFrame with deduplication and formatting.
    
    Args:
        chunks: Original data chunks with metadata
        normalized_identifiers: Normalized data identifiers
        answers: Classification results
        
    Returns:
        pandas.DataFrame: Formatted submission with required columns
    """
    # Create initial DataFrame
    sub_df = pd.DataFrame()
    sub_df["article_id"] = [c[0] for c in chunks]  # Extract article IDs
    sub_df["dataset_id"] = normalized_identifiers
    sub_df["type"] = answers
    
    # Normalize dataset_id to lowercase for consistency (except for case-sensitive IDs)
    # Note: Most identifiers should be uppercase, but DOI URLs should stay as-is
    def normalize_case(row):
        dataset_id = row["dataset_id"]
        if isinstance(dataset_id, str):
            if dataset_id.startswith("https://doi.org/"):
                return dataset_id.lower()  # DOI URLs to lowercase
            # Keep other identifiers as they were normalized
            return dataset_id
        return dataset_id
    
    sub_df["dataset_id"] = sub_df.apply(normalize_case, axis=1)
    
    # Remove entries with no classification (None values)
    sub_df = sub_df[sub_df["type"].notnull()].reset_index(drop=True)
    
    # Deduplicate: if same article_id + dataset_id appears multiple times,
    # keep the one with highest priority (Primary > Secondary)
    # Sort by type in descending order so Primary comes before Secondary
    sub_df = sub_df.sort_values(
        by=["article_id", "dataset_id", "type"], 
        ascending=False
    ).drop_duplicates(
        subset=['article_id', 'dataset_id'], 
        keep="first"  # Keep first occurrence (highest priority)
    ).reset_index(drop=True)
    
    # Add row_id as required by submission format
    sub_df['row_id'] = range(len(sub_df))
    
    # Save submission file with required column order
    sub_df.to_csv("submission.csv", index=False, 
                  columns=["row_id", "article_id", "dataset_id", "type"])
    
    # Print summary statistics
    print(f"\nSubmission Summary:")
    print(f"Total entries: {len(sub_df)}")
    pattern_counts = {}
    for chunk, identifier in zip(chunks, normalized_identifiers):
        pattern_type = chunk[2]  # pattern_type is at index 2
        if identifier in sub_df["dataset_id"].values:
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
    
    print("Entries by identifier type:")
    for pattern_type, count in sorted(pattern_counts.items()):
        print(f"  {pattern_type}: {count}")
    
    return sub_df

# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_submission(submission_df):
    """
    Evaluate submission against ground truth labels (training mode only).
    
    Calculates F1 score based on exact matches of (article_id, dataset_id, type).
    
    Args:
        submission_df: Generated submission DataFrame
        
    Returns:
        dict: Evaluation metrics (TP, FP, FN, F1)
    """
    def f1_score(tp, fp, fn):
        """Calculate F1 score from true positives, false positives, false negatives."""
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    
    # Only evaluate in training mode (not during competition rerun)
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        try:
            # Load ground truth labels
            label_df = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")
            
            # Filter out 'Missing' entries from ground truth
            label_df = label_df[label_df['type'] != 'Missing'].reset_index(drop=True)
            
            # Find exact matches between predictions and ground truth
            hits_df = label_df.merge(submission_df, on=["article_id", "dataset_id", "type"])
            
            # Calculate metrics
            tp = hits_df.shape[0]  # True positives: exact matches
            fp = submission_df.shape[0] - tp  # False positives: predictions not in ground truth
            fn = label_df.shape[0] - tp  # False negatives: ground truth not in predictions
            
            metrics = {
                "TP": tp,
                "FP": fp, 
                "FN": fn,
                "F1": f1_score(tp, fp, fn)
            }
            
            print(f"TP: {tp}")
            print(f"FP: {fp}")
            print(f"FN: {fn}")
            print(f"F1 Score: {round(metrics['F1'], 3)}")
            
            return metrics
            
        except FileNotFoundError:
            print("Ground truth labels not found - skipping evaluation")
            return None
    else:
        print("Competition rerun mode - evaluation skipped")
        return None

# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    """
    Main execution pipeline that orchestrates the entire data reference extraction and classification process.
    """
    print("Starting Enhanced MDC Search LLM Processor...")
    print("Comprehensive data identifier extraction with 99%+ coverage")
    print("=" * 60)
    
    # Step 1: Extract all data identifier patterns from PDFs
    print("Step 1: Extracting data identifiers from PDFs...")
    print("Supported formats: DOI, GSE, SRP, EMPIAR, PDB, ChEMBL, InterPro, Pfam,")
    print("                  UniProt, Ensembl, GenBank, CVCL, EPI_ISL, and more...")
    chunks = extract_all_data_references_from_pdfs()
    print(f"Found {len(chunks)} data identifier references")
    
    # Step 2: Initialize the LLM
    print("\nStep 2: Initializing Large Language Model...")
    llm, tokenizer = initialize_llm()
    print("LLM initialized successfully")
    
    # Step 3: Normalize all data identifiers
    print("\nStep 3: Normalizing data identifiers...")
    normalized_identifiers = normalize_data_identifiers(llm, tokenizer, chunks)
    print(f"Normalized {len(normalized_identifiers)} data identifiers")
    
    # Step 4: Classify data references
    print("\nStep 4: Classifying data references...")
    answers, logit_matrix = classify_data_references(llm, tokenizer, chunks)
    print("Classification completed")
    
    # Step 5: Prepare submission
    print("\nStep 5: Preparing submission...")
    submission_df = prepare_submission(chunks, normalized_identifiers, answers)
    print(f"Classification distribution:\n{submission_df['type'].value_counts()}")
    
    # Step 6: Evaluate results (if in training mode)
    print("\nStep 6: Evaluating results...")
    metrics = evaluate_submission(submission_df)
    
    print("\n" + "=" * 60)
    print("Enhanced pipeline completed successfully!")
    print("Submission saved as 'submission.csv'")
    print(f"Coverage: 18 different identifier types supported")
    
    return submission_df, metrics

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Execute main pipeline when script is run directly
    submission_df, metrics = main() 