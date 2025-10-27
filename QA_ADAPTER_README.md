# QA Dataset Adapter for DIFT

This module provides functionality to adapt Question-Answering (QA) datasets to the DIFT (Discrimination Instructions for Fine-Tuning) framework format.

## Overview

The `qa_data_construct.py` script converts standard QA datasets into the DIFT format, which includes the following fields:
- `input`: The formatted prompt with question and optional candidates
- `output`: The expected answer
- `query_id`: Unique identifier for each query (for knowledge injection)
- `entity_ids`: List of entity IDs corresponding to candidates (for knowledge injection)

## Input Format

The script accepts QA datasets in JSON format. Each entry should contain:

```json
{
    "question": "What is the capital of France?",
    "answer": "Paris",
    "candidates": ["Paris", "London", "Berlin", "Madrid"],  // optional
    "context": "Additional context information..."  // optional
}
```

### Flexible Field Names
The script supports flexible field names:
- Question: `question` or `input`
- Answer: `answer` or `output`
- Candidates: `candidates` (optional)
- Context: `context` (optional)

## Usage

### Basic Usage

```bash
python qa_data_construct.py \
    --llm_dir /path/to/llama-model \
    --input_file /path/to/qa_dataset.json \
    --output_dir /path/to/output
```

### Advanced Options

```bash
python qa_data_construct.py \
    --llm_dir /path/to/llama-model \
    --input_file /path/to/qa_dataset.json \
    --output_dir /path/to/output \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --add_special_tokens True \
    --add_context True \
    --seed 42
```

### Example with Sample Data

An example QA dataset is provided in `example_qa_dataset.json`. To test the adapter:

```bash
python qa_data_construct.py \
    --llm_dir llama-2-7b-chat-hf \
    --input_file example_qa_dataset.json \
    --output_dir ./qa_output
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | 42 | Random seed for reproducibility |
| `--llm_dir` | str | required | Path to LLM tokenizer directory |
| `--input_file` | str | required | Path to input QA dataset JSON file |
| `--output_dir` | str | required | Output directory for DIFT-formatted datasets |
| `--train_ratio` | float | 0.8 | Ratio of data for training set |
| `--val_ratio` | float | 0.1 | Ratio of data for validation set |
| `--add_special_tokens` | bool | True | Add [QUERY] and [ENTITY] tokens for knowledge injection |
| `--add_context` | bool | True | Include context in prompt if available |

## Output

The script generates the following files in the output directory:

1. **train.json**: Training set in DIFT format
2. **valid.json**: Validation set in DIFT format
3. **test.json**: Test set in DIFT format
4. **stats.json**: Dataset statistics including:
   - Number of examples in each split
   - Min/max/average sequence lengths

### Output Format Example

```json
{
    "input": "Question: What is the capital of France? [QUERY]\n\nSelect the best answer from the list: [Paris [ENTITY]; London [ENTITY]; Berlin [ENTITY]; Madrid [ENTITY]]\n\n[Answer]: ",
    "output": "Paris",
    "query_id": 0,
    "entity_ids": [0, 1, 2, 3],
    "candidates": ["Paris", "London", "Berlin", "Madrid"],
    "question": "What is the capital of France?"
}
```

## Integration with DIFT Training

After generating the DIFT-formatted QA dataset, you can use it for training with the existing DIFT framework:

### 1. Prepare Training Arguments

Create or modify the training script to use the QA dataset:

```bash
python train.py \
    --model_class LlamaForCausalLM \
    --model_name_or_path llama-2-7b-chat-hf \
    --dataset ./qa_output \
    --train_path ./qa_output/train.json \
    --eval_path ./qa_output/valid.json \
    --test_path ./qa_output/test.json \
    --output_dir ./output/qa_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 0.0002
```

### 2. Using with Knowledge Injection (KGELlama)

If you want to use the knowledge injection mechanism:

```bash
python train.py \
    --model_class KGELlama \
    --model_name_or_path llama-2-7b-chat-hf \
    --kge_model CoLE \
    --dataset ./qa_output \
    --train_path ./qa_output/train.json \
    --eval_path ./qa_output/valid.json \
    --test_path ./qa_output/test.json \
    --output_dir ./output/qa_kge_model
```

**Note**: When using KGELlama with QA datasets, you may need to prepare appropriate entity embeddings or use dummy embeddings if not working with a knowledge graph.

## Differences from Knowledge Graph Format

While the DIFT framework was originally designed for Knowledge Graph Completion, this adapter makes the following adaptations for QA tasks:

1. **Entity IDs**: For QA with candidates, entity_ids correspond to candidate indices. For open-ended QA, a dummy entity_id [0] is used.

2. **Query IDs**: Sequential IDs are assigned to each question rather than being based on KGE query embeddings.

3. **Special Tokens**: The [QUERY] token marks the question, and [ENTITY] tokens mark candidate answers (similar to how they mark entities in KG tasks).

4. **Prompt Format**: The prompt is adapted from KG triplet format to QA format while maintaining compatibility with DIFT's data collator.

## Extending the Adapter

To customize the adapter for specific QA datasets:

1. **Custom Prompt Format**: Modify the `create_prompt` method in `QADataAdapter` class
2. **Additional Fields**: Extend the `format_qa_to_dift` method to include more metadata
3. **Filtering**: Add filtering logic in the `load_qa_dataset` method

## Troubleshooting

### Issue: Tokenizer not found
**Solution**: Ensure the `--llm_dir` points to a valid Hugging Face model directory with tokenizer files.

### Issue: Out of memory during tokenization
**Solution**: Process the dataset in batches or reduce the sequence length in the tokenizer.

### Issue: Special tokens not recognized during training
**Solution**: Ensure the tokenizer in your training script adds the same special tokens: `tokenizer.add_tokens(['[QUERY]', '[ENTITY]', '[RELATION]'])`

## Citation

If you use this adapter, please cite the original DIFT paper:

```bibtex
@inproceedings{dift2024,
  title={Finetuning Generative Large Language Models with Discrimination Instructions for Knowledge Graph Completion},
  booktitle={ISWC},
  year={2024}
}
```
