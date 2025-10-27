import os
import json
import argparse
import random
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Optional
from transformers import AutoTokenizer


class QADataAdapter:
    """
    Adapter to convert QA datasets to DIFT format.
    
    DIFT format requirements:
    - input: The prompt/instruction text
    - output: The expected answer
    - query_id: Unique identifier for the query (for knowledge injection)
    - entity_ids: List of entity IDs (for knowledge injection)
    """
    
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def load_qa_dataset(self, file_path: str) -> List[Dict]:
        """
        Load QA dataset from JSON file.
        
        Expected input format (flexible):
        [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "candidates": ["Paris", "London", "Berlin", "Madrid"],  # optional
                "context": "...",  # optional
            },
            ...
        ]
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def format_qa_to_dift(self, qa_data: List[Dict]) -> List[Dict]:
        """
        Convert QA data to DIFT format.
        
        For each QA example, create a DIFT-compatible entry with:
        - input: Formatted prompt with candidates (if available)
        - output: The answer
        - query_id: Index of the query
        - entity_ids: Indices of candidate entities (or dummy values if not using KGE)
        """
        dift_data = []
        
        for idx, qa_item in enumerate(tqdm(qa_data, desc="Converting QA to DIFT format")):
            # Extract question and answer
            question = qa_item.get('question', qa_item.get('input', ''))
            answer = qa_item.get('answer', qa_item.get('output', ''))
            
            # Extract optional fields
            candidates = qa_item.get('candidates', [])
            context = qa_item.get('context', '')
            
            # Create the prompt
            prompt = self.create_prompt(question, answer, candidates, context)
            
            # Create entity_ids (for knowledge injection compatibility)
            # If candidates exist, use their indices; otherwise use dummy values
            if candidates:
                entity_ids = list(range(len(candidates)))
            else:
                # For open-ended QA without candidates, use a single dummy entity
                entity_ids = [0]
            
            # Create DIFT format entry
            dift_entry = {
                'input': prompt,
                'output': answer,
                'query_id': idx,
                'entity_ids': entity_ids,
            }
            
            # Preserve additional metadata if needed
            if candidates:
                dift_entry['candidates'] = candidates
            if 'question' in qa_item:
                dift_entry['question'] = question
            
            dift_data.append(dift_entry)
        
        return dift_data
    
    def create_prompt(self, question: str, answer: str, candidates: List[str], context: str) -> str:
        """
        Create a formatted prompt for the QA task.
        
        The prompt format is adapted from DIFT's knowledge graph prompts
        to work with QA tasks.
        """
        prompt = ""
        
        # Add context if available
        if context and self.args.add_context:
            prompt += f"Context:\n{context}\n\n"
        
        # Add the question
        if self.args.add_special_tokens and candidates:
            prompt += f"Question: {question} [QUERY]\n\n"
        else:
            prompt += f"Question: {question}\n\n"
        
        # Add candidates if available
        if candidates:
            if self.args.add_special_tokens:
                formatted_candidates = [f"{cand} [ENTITY]" for cand in candidates]
            else:
                formatted_candidates = candidates
            
            candidates_str = '[' + '; '.join(formatted_candidates) + ']'
            prompt += f"Select the best answer from the list: {candidates_str}\n\n"
        
        prompt += "[Answer]: "
        
        return prompt
    
    def split_dataset(self, data: List[Dict], train_ratio: float = 0.8, 
                     val_ratio: float = 0.1) -> tuple:
        """
        Split dataset into train, validation, and test sets.
        """
        random.shuffle(data)
        
        total = len(data)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data
    
    def save_dataset(self, data: List[Dict], output_path: str):
        """Save dataset to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(data)} examples to {output_path}")
    
    def compute_statistics(self, train_data: List[Dict], val_data: List[Dict], 
                          test_data: List[Dict]) -> Dict:
        """Compute and return dataset statistics."""
        stats = {
            'train_num': len(train_data),
            'valid_num': len(val_data),
            'test_num': len(test_data),
            'total_num': len(train_data) + len(val_data) + len(test_data),
        }
        
        # Compute average sequence lengths
        all_data = train_data + val_data + test_data
        texts = [item['input'] for item in all_data]
        encoded = self.tokenizer(texts, add_special_tokens=False)
        lens = [len(input_ids) for input_ids in encoded['input_ids']]
        
        stats['min_seq_len'] = int(np.min(lens))
        stats['max_seq_len'] = int(np.max(lens))
        stats['avg_seq_len'] = int(np.round(np.mean(lens)))
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Convert QA datasets to DIFT format")
    
    # Basic arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--llm_dir', type=str, required=True, 
                       help='Path to LLM tokenizer directory')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input QA dataset JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for DIFT-formatted datasets')
    
    # Data processing arguments
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of validation data')
    parser.add_argument('--add_special_tokens', type=bool, default=True,
                       help='Add [QUERY] and [ENTITY] tokens for knowledge injection')
    parser.add_argument('--add_context', type=bool, default=True,
                       help='Include context in the prompt if available')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize adapter
    print(f"Initializing QA Data Adapter...")
    adapter = QADataAdapter(args)
    
    # Load QA dataset
    print(f"Loading QA dataset from {args.input_file}...")
    qa_data = adapter.load_qa_dataset(args.input_file)
    print(f"Loaded {len(qa_data)} QA examples")
    
    # Convert to DIFT format
    print("Converting to DIFT format...")
    dift_data = adapter.format_qa_to_dift(qa_data)
    
    # Split dataset
    print("Splitting dataset...")
    train_data, val_data, test_data = adapter.split_dataset(
        dift_data, args.train_ratio, args.val_ratio
    )
    
    # Save datasets
    print("Saving datasets...")
    adapter.save_dataset(train_data, os.path.join(args.output_dir, 'train.json'))
    adapter.save_dataset(val_data, os.path.join(args.output_dir, 'valid.json'))
    adapter.save_dataset(test_data, os.path.join(args.output_dir, 'test.json'))
    
    # Compute and save statistics
    print("Computing statistics...")
    stats = adapter.compute_statistics(train_data, val_data, test_data)
    stats_path = os.path.join(args.output_dir, 'stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
    
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nQA dataset successfully adapted to DIFT format!")
    print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
