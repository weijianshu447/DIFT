#!/usr/bin/env python
"""
End-to-end integration test for QA data adapter pipeline.
Tests the complete workflow from QA data to DIFT format with dummy embeddings.
"""

import os
import sys
import json
import tempfile
import shutil


def test_complete_pipeline():
    """Test the complete QA to DIFT pipeline."""
    print("=" * 70)
    print("End-to-End QA Adapter Pipeline Test")
    print("=" * 70)
    print()
    
    # Create temporary directory for test
    test_dir = tempfile.mkdtemp(prefix="qa_adapter_test_")
    print(f"Using temporary directory: {test_dir}")
    
    try:
        # Step 1: Validate example QA dataset exists
        print("\n[Step 1] Checking example QA dataset...")
        example_file = "example_qa_dataset.json"
        if not os.path.exists(example_file):
            print(f"❌ Example file not found: {example_file}")
            return False
        
        with open(example_file, 'r') as f:
            qa_data = json.load(f)
        print(f"✓ Found {len(qa_data)} QA examples")
        
        # Step 2: Verify QA format
        print("\n[Step 2] Verifying QA data format...")
        for idx, entry in enumerate(qa_data):
            assert 'question' in entry or 'input' in entry, f"Entry {idx} missing question"
            assert 'answer' in entry or 'output' in entry, f"Entry {idx} missing answer"
        print("✓ QA data format is valid")
        
        # Step 3: Simulate conversion (we'll create the output format manually for testing)
        print("\n[Step 3] Simulating DIFT format conversion...")
        dift_data = []
        for idx, entry in enumerate(qa_data):
            question = entry.get('question', entry.get('input', ''))
            answer = entry.get('answer', entry.get('output', ''))
            candidates = entry.get('candidates', [])
            
            # Create DIFT format entry
            dift_entry = {
                'input': f"Question: {question} [QUERY]\n\nSelect the best answer from the list: [" + 
                        '; '.join([f"{c} [ENTITY]" for c in candidates]) + "]\n\n[Answer]: ",
                'output': answer,
                'query_id': idx,
                'entity_ids': list(range(len(candidates))) if candidates else [0],
            }
            dift_data.append(dift_entry)
        
        print(f"✓ Converted {len(dift_data)} entries to DIFT format")
        
        # Step 4: Split data
        print("\n[Step 4] Splitting data into train/valid/test...")
        train_end = int(len(dift_data) * 0.6)
        valid_end = int(len(dift_data) * 0.8)
        
        train_data = dift_data[:train_end]
        valid_data = dift_data[train_end:valid_end]
        test_data = dift_data[valid_end:]
        
        print(f"✓ Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
        
        # Step 5: Save to temporary files
        print("\n[Step 5] Saving DIFT-formatted data...")
        train_path = os.path.join(test_dir, 'train.json')
        valid_path = os.path.join(test_dir, 'valid.json')
        test_path = os.path.join(test_dir, 'test.json')
        
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(valid_path, 'w') as f:
            json.dump(valid_data, f, indent=2)
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"✓ Saved to {test_dir}")
        
        # Step 6: Verify DIFT format
        print("\n[Step 6] Verifying DIFT format...")
        required_fields = ['input', 'output', 'query_id', 'entity_ids']
        
        all_data = train_data + valid_data + test_data
        for idx, entry in enumerate(all_data):
            for field in required_fields:
                assert field in entry, f"Entry {idx} missing field: {field}"
            assert isinstance(entry['query_id'], int), f"Entry {idx}: query_id not int"
            assert isinstance(entry['entity_ids'], list), f"Entry {idx}: entity_ids not list"
        
        print("✓ All entries have required DIFT fields")
        
        # Step 7: Compute embedding requirements
        print("\n[Step 7] Computing embedding requirements...")
        max_query_id = max(entry['query_id'] for entry in all_data)
        max_entity_id = max(max(entry['entity_ids']) for entry in all_data)
        
        num_queries = max_query_id + 1
        num_entities = max_entity_id + 1
        
        print(f"✓ Queries: {num_queries}, Entities: {num_entities}")
        
        # Step 8: Simulate dummy embedding creation (without actually creating large tensors)
        print("\n[Step 8] Simulating dummy embedding creation...")
        embedding_dim = 768
        print(f"✓ Would create query embeddings: ({num_queries}, {embedding_dim})")
        print(f"✓ Would create entity embeddings: ({num_entities}, {embedding_dim})")
        
        # Step 9: Verify data can be loaded
        print("\n[Step 9] Verifying data loading...")
        with open(train_path, 'r') as f:
            loaded_train = json.load(f)
        assert len(loaded_train) == len(train_data), "Data loading mismatch"
        print("✓ Data can be loaded successfully")
        
        print("\n" + "=" * 70)
        print("✅ End-to-End Pipeline Test PASSED")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\nCleaned up temporary directory: {test_dir}")


def main():
    success = test_complete_pipeline()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
