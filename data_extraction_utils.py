#!/usr/bin/env python3
"""
Data Extraction Utilities Module

This module contains common functions for extracting data identifiers from academic papers.
Designed to be imported by both the main LLM processor and test scripts.

Key Features:
- Comprehensive pattern matching for 18+ identifier types
- PDF text extraction with reference section filtering
- Identifier normalization and validation
- Repository-specific pattern handling
"""

import os
import re
from typing import List, Tuple, Dict, Set
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not available. PDF processing will be disabled.")

def get_comprehensive_patterns() -> Dict[str, re.Pattern]:
    """
    Get comprehensive regex patterns for all data identifier types.
    
    Based on analysis of training labels achieving 99.5+ % coverage of identifier types.
    
    Returns:
        Dict[str, re.Pattern]: Dictionary mapping pattern names to compiled regex patterns
    """
    patterns = {
        # DOI patterns
        'doi_https': re.compile(r'https://doi\.org/10\.\d+/[^\s,;)\]]+', re.IGNORECASE),
        'doi_bare': re.compile(r'10\.\d+/[^\s,;)\]]+', re.IGNORECASE),
        
        # Gene Expression Omnibus and related
        'gse': re.compile(r'GSE\d+', re.IGNORECASE),
        'srp': re.compile(r'SRP\d+', re.IGNORECASE),
        'arrayexpress_geod': re.compile(r'E-GEOD-\d+', re.IGNORECASE),
        'arrayexpress_general': re.compile(r'E-[A-Z]{4}-\d+', re.IGNORECASE),
        
        # Structural biology
        'empiar': re.compile(r'EMPIAR-\d+', re.IGNORECASE),
        'pdb_upper': re.compile(r'[0-9][A-Z0-9]{3}', re.IGNORECASE),
        'pdb_lower': re.compile(r'[0-9][a-z0-9]{3}', re.IGNORECASE),
        'cath': re.compile(r'\d+\.\d+\.\d+\.\d+', re.IGNORECASE),
        
        # Chemical databases
        'chembl': re.compile(r'CHEMBL\d+', re.IGNORECASE),
        
        # Protein databases
        'ipr': re.compile(r'IPR\d+', re.IGNORECASE),
        'pfam': re.compile(r'PF\d+', re.IGNORECASE),
        'uniprot': re.compile(r'[A-Z]\d[A-Z0-9]{3}\d|[OPQ][0-9][A-Z0-9]{3}[0-9]', re.IGNORECASE),
        
        # Genomics
        'ensembl': re.compile(r'ENS[A-Z]+\d+', re.IGNORECASE),
        'genbank': re.compile(r'[A-Z]{1,2}\d+(?:\.\d+)?', re.IGNORECASE),
        
        # Cell lines and epidemiological data
        'cell_lines': re.compile(r'CVCL_\w+', re.IGNORECASE),
        'epi_isl': re.compile(r'EPI_ISL_\d+', re.IGNORECASE),
        'other_db': re.compile(r'[A-Z]{3,}[0-9]+', re.IGNORECASE),
    }
    
    return patterns

def normalize_identifier(raw_id: str, pattern_type: str) -> str:
    """
    Normalize different types of identifiers according to submission requirements.
    
    Args:
        raw_id: Raw identifier string
        pattern_type: Type of pattern (from patterns dict)
        
    Returns:
        str: Normalized identifier
    """
    raw_id = str(raw_id).strip()
    
    # DOI patterns - normalize to full https://doi.org/ format as required
    if pattern_type == 'doi_https':
        return raw_id  # Already in correct format
    elif pattern_type == 'doi_bare':
        return f"https://doi.org/{raw_id}"
    
    # All other identifiers - return as-is but clean up
    else:
        # Convert to uppercase for consistency (except lowercase PDB codes)
        if pattern_type == 'pdb_lower':
            return raw_id.lower()
        else:
            return raw_id.upper()

def extract_text_from_pdf(pdf_path: str, stop_at_references: bool = True) -> str:
    """
    Extract text from a PDF file, optionally stopping at the references section.
    
    Args:
        pdf_path: Path to the PDF file
        stop_at_references: If True, stop extraction at 'references' section
        
    Returns:
        str: Extracted text content
        
    Raises:
        ImportError: If PyMuPDF is not available
        Exception: If PDF cannot be processed
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF is required for PDF processing")
    
    doc = fitz.open(pdf_path)
    text = ""
    
    try:
        for page in doc:
            page_text = page.get_text().lower()
            
            if stop_at_references and 'references' in page_text:
                # Stop processing if we reach the references section
                # (identifiers in references are typically not data references)
                page_text = page_text.split("references")[0]
                text += page_text
                break
            else:
                text += page_text
    finally:
        doc.close()
    
    return text

def extract_identifiers_from_text(
    text: str, 
    patterns: Dict[str, re.Pattern], 
    article_id: str,
    text_span_len: int = 100
) -> List[Tuple[str, str, str, str]]:
    """
    Extract data identifiers from text using provided patterns.
    
    Args:
        text: Text content to search
        patterns: Dictionary of compiled regex patterns
        article_id: Article identifier to exclude self-references
        text_span_len: Length of context to extract around each match
        
    Returns:
        List[Tuple[str, str, str, str]]: List of (article_id, context, pattern_type, raw_identifier)
    """
    results = []
    
    # Search for all pattern types in the text
    for pattern_name, pattern in patterns.items():
        matches = pattern.finditer(text)
        
        for match in matches:
            raw_identifier = match.group()
            
            # Skip if this identifier is part of the article's own ID
            if raw_identifier in article_id:
                continue
            
            # Extract text chunk around the identifier for context
            chunk_start = max(0, match.start() - text_span_len)
            chunk_end = match.start() + text_span_len
            context = text[chunk_start:chunk_end]
            
            results.append((article_id, context, pattern_name, raw_identifier))
    
    return results

def extract_data_references_from_pdfs(
    pdf_directory: str,
    patterns: Dict[str, re.Pattern] = None,
    text_span_len: int = 100,
    stop_at_references: bool = True
) -> List[Tuple[str, str, str, str]]:
    """
    Extract data identifiers from all PDFs in a directory.
    
    Args:
        pdf_directory: Directory containing PDF files
        patterns: Dictionary of regex patterns (uses default if None)
        text_span_len: Length of context to extract around each match
        stop_at_references: Whether to stop extraction at references section
        
    Returns:
        List[Tuple[str, str, str, str]]: List of (article_id, context, pattern_type, raw_identifier)
    """
    if patterns is None:
        patterns = get_comprehensive_patterns()
    
    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    all_results = []
    
    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_directory, filename)
        article_id = filename.split(".pdf")[0]
        
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path, stop_at_references)
            
            # Extract identifiers from text
            results = extract_identifiers_from_text(
                text, patterns, article_id, text_span_len
            )
            
            all_results.extend(results)
            
        except Exception as e:
            continue
    
    return all_results

def get_kaggle_pdf_directory() -> str:
    """
    Get the appropriate PDF directory path for Kaggle environment.
    
    Returns:
        str: Path to PDF directory based on environment
    """
    # Determine PDF directory based on environment (Kaggle competition vs local)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        return "/kaggle/input/make-data-count-finding-data-references/test/PDF"
    else:
        return "/kaggle/input/make-data-count-finding-data-references/train/PDF"

def validate_identifier_format(identifier: str, pattern_type: str) -> bool:
    """
    Validate that an identifier matches expected format for its type.
    
    Args:
        identifier: The identifier to validate
        pattern_type: The type of pattern
        
    Returns:
        bool: True if identifier is valid for the pattern type
    """
    # Basic format validation
    if not identifier or len(identifier.strip()) < 3:
        return False
    
    # Type-specific validation
    if pattern_type == 'doi_https':
        return identifier.startswith('https://doi.org/10.')
    elif pattern_type == 'doi_bare':
        return identifier.startswith('10.')
    elif pattern_type == 'gse':
        return identifier.upper().startswith('GSE') and len(identifier) >= 6
    elif pattern_type == 'empiar':
        return identifier.upper().startswith('EMPIAR-')
    
    # Default validation - just check it's not empty
    return True

def filter_common_false_positives(
    results: List[Tuple[str, str, str, str]]
) -> List[Tuple[str, str, str, str]]:
    """
    Filter out common false positive patterns.
    
    Args:
        results: List of extraction results
        
    Returns:
        List[Tuple[str, str, str, str]]: Filtered results
    """
    filtered_results = []
    
    # Common false positive patterns to exclude
    false_positive_contexts = [
        r'figure\s+\d+',
        r'fig\.\s*\d+',
        r'table\s+\d+',
        r'tab\.\s*\d+',
        r'protocol\s+',
        r'version\s+',
        r'approval\s+',
    ]
    
    false_positive_pattern = re.compile('|'.join(false_positive_contexts), re.IGNORECASE)
    
    for article_id, context, pattern_type, raw_identifier in results:
        # Check if context contains false positive indicators
        if false_positive_pattern.search(context):
            continue
        
        # Check if identifier is valid for its type
        if not validate_identifier_format(raw_identifier, pattern_type):
            continue
        
        filtered_results.append((article_id, context, pattern_type, raw_identifier))
    
    return filtered_results

# Example usage and testing
if __name__ == "__main__":
    print("Data Extraction Utils Module")
    print("=" * 40)
    
    # Test pattern compilation
    patterns = get_comprehensive_patterns()
    print(f"Loaded {len(patterns)} identifier patterns:")
    for name in patterns.keys():
        print(f"  - {name}")
    
    # Test normalization
    test_cases = [
        ("10.5061/dryad.123", "doi_bare"),
        ("GSE12345", "gse"),
        ("5va1", "pdb_lower"),
        ("CHEMBL123456", "chembl")
    ]
    
    print(f"\nNormalization tests:")
    for raw_id, pattern_type in test_cases:
        normalized = normalize_identifier(raw_id, pattern_type)
        print(f"  {raw_id} ({pattern_type}) -> {normalized}")
    
    print(f"\nPyMuPDF available: {PYMUPDF_AVAILABLE}") 