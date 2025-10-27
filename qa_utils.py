"""
Helper functions for working with QA datasets in DIFT framework.
"""

import torch
import os
import json
from typing import List, Dict


def create_dummy_embeddings(num_queries: int, num_entities: int, 
                           embedding_dim: int = 768, 
                           output_dir: str = None,
                           seed: int = 42):
    """
    Create dummy embeddings for QA datasets that don't have KGE embeddings.
    
    This is useful when you want to use the KGELlama model architecture
    but don't have pre-trained KGE embeddings for your QA dataset.
    
    Args:
        num_queries: Number of unique queries in the dataset
        num_entities: Number of unique entities (candidates) in the dataset
        embedding_dim: Dimension of the embeddings (default: 768 for BERT-like models)
        output_dir: Directory to save the embeddings. If None, returns the tensors.
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        If output_dir is None: tuple of (query_embeddings, entity_embeddings)
        Otherwise: None (saves to disk)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Initialize random embeddings
    query_embeddings = torch.randn(num_queries, embedding_dim)
    entity_embeddings = torch.randn(num_entities, embedding_dim)
    
    # Normalize embeddings
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    entity_embeddings = torch.nn.functional.normalize(entity_embeddings, p=2, dim=1)
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(query_embeddings, os.path.join(output_dir, 'query_embeddings.pt'))
        torch.save(entity_embeddings, os.path.join(output_dir, 'entity_embeddings.pt'))
        print(f"Saved embeddings to {output_dir}")
        print(f"  - query_embeddings.pt: {query_embeddings.shape}")
        print(f"  - entity_embeddings.pt: {entity_embeddings.shape}")
        return None
    else:
        return query_embeddings, entity_embeddings


def compute_qa_embedding_requirements(data_files: List[str]) -> Dict:
    """
    Compute the embedding requirements for QA dataset files.
    
    Args:
        data_files: List of paths to QA dataset JSON files (train, valid, test)
    
    Returns:
        Dictionary with 'num_queries' and 'num_entities'
    """
    max_query_id = -1
    max_entity_id = -1
    
    for file_path in data_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for entry in data:
            query_id = entry.get('query_id', 0)
            entity_ids = entry.get('entity_ids', [0])
            
            max_query_id = max(max_query_id, query_id)
            max_entity_id = max(max_entity_id, max(entity_ids))
    
    return {
        'num_queries': max_query_id + 1,
        'num_entities': max_entity_id + 1,
    }


def prepare_qa_embeddings(train_path: str, valid_path: str, test_path: str,
                         output_dir: str, embedding_dim: int = 768, seed: int = 42):
    """
    Prepare dummy embeddings for a QA dataset.
    
    This is a convenience function that:
    1. Analyzes the QA dataset to determine embedding requirements
    2. Creates and saves dummy embeddings
    
    Args:
        train_path: Path to training data JSON file
        valid_path: Path to validation data JSON file
        test_path: Path to test data JSON file
        output_dir: Directory to save the embeddings
        embedding_dim: Dimension of the embeddings
        seed: Random seed for reproducibility
    """
    print("Analyzing QA dataset...")
    requirements = compute_qa_embedding_requirements([train_path, valid_path, test_path])
    
    print(f"Dataset requirements:")
    print(f"  - Number of queries: {requirements['num_queries']}")
    print(f"  - Number of entities: {requirements['num_entities']}")
    
    print(f"\nCreating dummy embeddings with dim={embedding_dim}...")
    create_dummy_embeddings(
        num_queries=requirements['num_queries'],
        num_entities=requirements['num_entities'],
        embedding_dim=embedding_dim,
        output_dir=output_dir,
        seed=seed
    )
    
    print(f"\n✓ QA embeddings prepared successfully!")
    print(f"Use --kge_model QA_embeddings when training")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dummy embeddings for QA datasets")
    parser.add_argument('--train_path', type=str, required=True, help='Path to train.json')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to valid.json')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test.json')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Output directory for embeddings')
    parser.add_argument('--embedding_dim', type=int, default=768,
                       help='Embedding dimension (default: 768)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    prepare_qa_embeddings(
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        seed=args.seed
    )
