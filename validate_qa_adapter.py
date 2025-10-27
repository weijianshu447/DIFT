#!/usr/bin/env python
"""
Validation script for QA data adapter.
Tests the QA data conversion without requiring a full LLM tokenizer.
"""

import json
import sys
import os

def validate_qa_input(file_path):
    """Validate input QA dataset format."""
    print(f"Validating input file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return False
    
    if not isinstance(data, list):
        print("❌ Data should be a list of QA entries")
        return False
    
    print(f"✓ Found {len(data)} QA entries")
    
    required_fields = ['question', 'answer']
    optional_fields = ['candidates', 'context', 'input', 'output']
    
    for idx, entry in enumerate(data):
        # Check if it has either 'question' or 'input'
        has_question = 'question' in entry or 'input' in entry
        has_answer = 'answer' in entry or 'output' in entry
        
        if not has_question:
            print(f"❌ Entry {idx} missing 'question' or 'input' field")
            return False
        
        if not has_answer:
            print(f"❌ Entry {idx} missing 'answer' or 'output' field")
            return False
    
    print("✓ All entries have required fields")
    
    # Check for candidates
    with_candidates = sum(1 for entry in data if 'candidates' in entry)
    print(f"✓ {with_candidates}/{len(data)} entries have candidate answers")
    
    # Check for context
    with_context = sum(1 for entry in data if 'context' in entry)
    print(f"✓ {with_context}/{len(data)} entries have context")
    
    return True


def validate_dift_output(file_path):
    """Validate DIFT-formatted output."""
    print(f"\nValidating DIFT output file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return False
    
    if not isinstance(data, list):
        print("❌ Data should be a list")
        return False
    
    print(f"✓ Found {len(data)} entries")
    
    required_fields = ['input', 'output', 'query_id', 'entity_ids']
    
    for idx, entry in enumerate(data):
        for field in required_fields:
            if field not in entry:
                print(f"❌ Entry {idx} missing required field: {field}")
                return False
        
        # Validate types
        if not isinstance(entry['query_id'], int):
            print(f"❌ Entry {idx}: query_id should be int")
            return False
        
        if not isinstance(entry['entity_ids'], list):
            print(f"❌ Entry {idx}: entity_ids should be list")
            return False
    
    print("✓ All entries have required DIFT fields")
    print("✓ All field types are correct")
    
    return True


def main():
    print("=" * 60)
    print("QA Data Adapter Validation Script")
    print("=" * 60)
    
    # Check if example file exists
    example_file = "example_qa_dataset.json"
    if not os.path.exists(example_file):
        print(f"❌ Example file not found: {example_file}")
        sys.exit(1)
    
    # Validate input format
    if not validate_qa_input(example_file):
        print("\n❌ Validation failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Input validation passed!")
    print("=" * 60)
    
    # Check if output exists (if conversion has been run)
    output_files = ['./qa_output/train.json', './qa_output/valid.json', './qa_output/test.json']
    
    any_output_exists = any(os.path.exists(f) for f in output_files)
    
    if any_output_exists:
        print("\nValidating output files...")
        all_valid = True
        for output_file in output_files:
            if os.path.exists(output_file):
                if not validate_dift_output(output_file):
                    all_valid = False
        
        if all_valid:
            print("\n" + "=" * 60)
            print("✅ All output files are valid!")
            print("=" * 60)
        else:
            print("\n❌ Some output files failed validation")
            sys.exit(1)
    else:
        print("\nℹ️  No output files found. Run the adapter first:")
        print("   python qa_data_construct.py --llm_dir <path> --input_file example_qa_dataset.json --output_dir ./qa_output")
    
    print("\n✅ Validation complete!")


if __name__ == '__main__':
    main()
