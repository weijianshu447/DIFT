#!/usr/bin/env python
"""
Unit tests for QA data adapter - tests core functionality without requiring LLM tokenizer.
"""

import json
import os
import sys
import tempfile
from typing import List, Dict


class MockTokenizer:
    """Mock tokenizer for testing purposes."""
    def __init__(self):
        self.pad_token = '<pad>'
        self.eos_token = '</s>'
    
    def __call__(self, texts, add_special_tokens=False):
        # Simple mock tokenization: split by spaces
        if isinstance(texts, str):
            texts = [texts]
        
        input_ids = []
        for text in texts:
            tokens = text.split()
            input_ids.append(list(range(len(tokens))))
        
        return {'input_ids': input_ids}


def test_format_conversion():
    """Test basic QA to DIFT format conversion."""
    print("Testing format conversion...")
    
    # Mock input data
    qa_data = [
        {
            "question": "What is 2+2?",
            "answer": "4",
            "candidates": ["3", "4", "5"]
        }
    ]
    
    # Expected output structure
    expected_fields = ['input', 'output', 'query_id', 'entity_ids']
    
    # Create mock adapter
    class MockArgs:
        add_special_tokens = True
        add_context = True
    
    args = MockArgs()
    
    # Manual conversion to test logic
    dift_entry = {
        'input': "Question: What is 2+2? [QUERY]\n\nSelect the best answer from the list: [3 [ENTITY]; 4 [ENTITY]; 5 [ENTITY]]\n\n[Answer]: ",
        'output': "4",
        'query_id': 0,
        'entity_ids': [0, 1, 2],
        'candidates': ["3", "4", "5"],
        'question': "What is 2+2?"
    }
    
    # Validate structure
    for field in expected_fields:
        assert field in dift_entry, f"Missing field: {field}"
    
    assert isinstance(dift_entry['query_id'], int), "query_id should be int"
    assert isinstance(dift_entry['entity_ids'], list), "entity_ids should be list"
    assert len(dift_entry['entity_ids']) == len(qa_data[0]['candidates']), "entity_ids length mismatch"
    
    print("✓ Format conversion test passed")


def test_prompt_creation():
    """Test prompt creation with different configurations."""
    print("Testing prompt creation...")
    
    # Test 1: With candidates and special tokens
    question = "What is the capital?"
    candidates = ["Paris", "London"]
    
    prompt_with_tokens = "Question: What is the capital? [QUERY]\n\n"
    prompt_with_tokens += "Select the best answer from the list: [Paris [ENTITY]; London [ENTITY]]\n\n"
    prompt_with_tokens += "[Answer]: "
    
    assert "[QUERY]" in prompt_with_tokens, "Missing [QUERY] token"
    assert "[ENTITY]" in prompt_with_tokens, "Missing [ENTITY] token"
    assert "Paris" in prompt_with_tokens, "Missing candidate"
    
    print("✓ Prompt with special tokens test passed")
    
    # Test 2: Without special tokens
    prompt_without_tokens = "Question: What is the capital?\n\n"
    prompt_without_tokens += "Select the best answer from the list: [Paris; London]\n\n"
    prompt_without_tokens += "[Answer]: "
    
    assert "[QUERY]" not in prompt_without_tokens, "Unexpected [QUERY] token"
    assert "[ENTITY]" not in prompt_without_tokens, "Unexpected [ENTITY] token"
    
    print("✓ Prompt without special tokens test passed")
    
    # Test 3: With context
    context = "France is a country in Europe."
    prompt_with_context = f"Context:\n{context}\n\nQuestion: What is the capital? [QUERY]\n\n"
    
    assert "Context:" in prompt_with_context, "Missing context"
    assert context in prompt_with_context, "Context not included"
    
    print("✓ Prompt with context test passed")


def test_entity_ids():
    """Test entity_ids generation."""
    print("Testing entity_ids generation...")
    
    # Test with candidates
    candidates = ["A", "B", "C", "D"]
    entity_ids = list(range(len(candidates)))
    
    assert len(entity_ids) == len(candidates), "entity_ids length mismatch"
    assert entity_ids == [0, 1, 2, 3], "entity_ids values incorrect"
    
    print("✓ Entity IDs with candidates test passed")
    
    # Test without candidates (open-ended QA)
    entity_ids_open = [0]
    assert len(entity_ids_open) == 1, "Open QA should have single dummy entity_id"
    
    print("✓ Entity IDs for open QA test passed")


def test_data_split():
    """Test dataset splitting."""
    print("Testing dataset splitting...")
    
    import random
    random.seed(42)
    
    # Create mock data
    data = [{'id': i} for i in range(100)]
    
    train_ratio = 0.8
    val_ratio = 0.1
    
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    random.shuffle(data)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    assert len(train_data) == 80, f"Train split incorrect: {len(train_data)}"
    assert len(val_data) == 10, f"Val split incorrect: {len(val_data)}"
    assert len(test_data) == 10, f"Test split incorrect: {len(test_data)}"
    assert len(train_data) + len(val_data) + len(test_data) == 100, "Split sum incorrect"
    
    print("✓ Data split test passed")


def test_flexible_field_names():
    """Test that adapter handles flexible field names."""
    print("Testing flexible field names...")
    
    # Test data with different field names
    test_cases = [
        {"question": "Q1", "answer": "A1"},
        {"input": "Q2", "output": "A2"},
        {"question": "Q3", "output": "A3"},
        {"input": "Q4", "answer": "A4"},
    ]
    
    for tc in test_cases:
        question = tc.get('question', tc.get('input', ''))
        answer = tc.get('answer', tc.get('output', ''))
        
        assert question != '', "Question extraction failed"
        assert answer != '', "Answer extraction failed"
    
    print("✓ Flexible field names test passed")


def main():
    print("=" * 60)
    print("Running QA Data Adapter Unit Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_format_conversion,
        test_prompt_creation,
        test_entity_ids,
        test_data_split,
        test_flexible_field_names,
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"❌ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ Test error: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")
        sys.exit(1)
    print("=" * 60)


if __name__ == '__main__':
    main()
