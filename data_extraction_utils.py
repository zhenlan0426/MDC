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
        List[Tuple[str, str, str, str]]: List of (article_id, context, pattern_type, normalized_identifier)
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
            
            # Normalize the identifier
            normalized_identifier = normalize_identifier(raw_identifier, pattern_name)
            
            # Extract text chunk around the identifier for context
            chunk_start = max(0, match.start() - text_span_len)
            chunk_end = match.start() + text_span_len
            context = text[chunk_start:chunk_end]
            
            results.append((article_id, context, pattern_name, normalized_identifier))
    
    return results

def extract_data_references_from_pdfs(
    pdf_directory: str,
    patterns: Dict[str, re.Pattern] = None,
    text_span_len: int = 100,
    stop_at_references: bool = True,
    apply_false_positive_filtering: bool = True
) -> List[Tuple[str, str, str, str]]:
    """
    Extract data identifiers from all PDFs in a directory.
    
    Args:
        pdf_directory: Directory containing PDF files
        patterns: Dictionary of regex patterns (uses default if None)
        text_span_len: Length of context to extract around each match
        stop_at_references: Whether to stop extraction at references section
        apply_false_positive_filtering: Whether to apply false positive filtering
        
    Returns:
        List[Tuple[str, str, str, str]]: List of (article_id, context, pattern_type, normalized_identifier)
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
    
    # Apply improved false positive filtering to all results
    if apply_false_positive_filtering:
        all_results = filter_common_false_positives(all_results)
    
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

def validate_identifier_content(identifier: str, pattern_type: str) -> bool:
    """Additional validation for identifier content."""
    # Skip very short identifiers (likely false positives)
    if len(identifier) < 4:
        return False
    
    # Skip pure numeric identifiers that look like years
    if identifier.isdigit() and 1900 <= int(identifier) <= 2100:
        return False
    
    # For PDB patterns, ensure they're not just years or common false positives
    if pattern_type in ['pdb_upper', 'pdb_lower']:
        if identifier.isdigit():
            return False
        # Should have at least one letter for valid PDB codes
        if not re.search(r'[a-zA-Z]', identifier):
            return False
    
    # Skip chemical formulas that are too generic
    if re.match(r'^[CHNOSPchnosp]+\d*$', identifier) and len(identifier) < 6:
        return False
    
    return True

def filter_common_false_positives(
    results: List[Tuple[str, str, str, str]]
) -> List[Tuple[str, str, str, str]]:
    """
    Filter out common false positive patterns using improved filtering logic.
    
    Args:
        results: List of extraction results with normalized identifiers
        
    Returns:
        List[Tuple[str, str, str, str]]: Filtered results
    """
    filtered_results = []
    
    # Improved false positive patterns
    false_positive_patterns = [
        r'\b(19|20)\d{2}\b',  # Years (1900-2099)
        r'\b\d{4}\b',  # Generic 4-digit numbers that might be years
        r'\b[0-9]+[a-z]\d+\b',  # Chemical formulas like c24h32
        r'\bfig\w*\s*\d+',  # Figure references
        r'\btab\w*\s*\d+',  # Table references
        r'\bp\s*[<>=]\s*0\.\d+',  # P-values
        r'\bn\s*=\s*\d+',  # Sample sizes
        r'\bph\s*\d+\.\d+',  # pH values
        r'\bic50\b',  # IC50 values (not database identifiers)
        r'\bf\d+\b',  # F-statistics
        r'\bt\d+\b',  # T-statistics
        r'\br\d+\b',  # R-values/correlation coefficients
    ]
    
    false_positive_regex = re.compile('|'.join(false_positive_patterns), re.IGNORECASE)
    
    for article_id, context, pattern_type, normalized_identifier in results:
        # Skip if identifier matches false positive patterns
        if false_positive_regex.search(normalized_identifier):
            continue
        
        # Skip if context suggests this is not a data reference
        context_lower = context.lower()
        if any(phrase in context_lower for phrase in [
            'figure', 'fig.', 'table', 'tab.', 'protocol', 'version',
            'approval', 'equation', 'formula', 'temperature', 'concentration'
        ]):
            continue
        
        # For DOI patterns, only keep those that look like real DOIs
        if pattern_type in ['doi_https', 'doi_bare']:
            if 'doi.org' in normalized_identifier or normalized_identifier.startswith('10.'):
                # Check if it's a reasonable DOI format
                if re.match(r'10\.\d+/\S+', normalized_identifier.replace('https://doi.org/', '')):
                    filtered_results.append((article_id, context, pattern_type, normalized_identifier))
            continue
        
        # For other patterns, apply additional validation
        if validate_identifier_content(normalized_identifier, pattern_type):
            # Check if identifier is valid for its type
            if validate_identifier_format(normalized_identifier, pattern_type):
                filtered_results.append((article_id, context, pattern_type, normalized_identifier))
    
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