# QA Adapter Quick Start Guide

## 1-Minute Setup

### Step 1: Validate Your QA Data
```bash
# Check if your QA dataset is in the correct format
python validate_qa_adapter.py
```

### Step 2: Convert to DIFT Format
```bash
python qa_data_construct.py \
    --llm_dir /path/to/llama-2-7b-chat-hf \
    --input_file your_qa_data.json \
    --output_dir ./qa_dift_output
```

### Step 3: (Optional) Prepare Embeddings for Knowledge Injection
```bash
python qa_utils.py \
    --train_path ./qa_dift_output/train.json \
    --valid_path ./qa_dift_output/valid.json \
    --test_path ./qa_dift_output/test.json \
    --output_dir ./qa_dift_output/embeddings
```

### Step 4: Train
```bash
# Without knowledge injection
python train.py \
    --model_class LlamaForCausalLM \
    --model_name_or_path /path/to/llama-2-7b-chat-hf \
    --train_path ./qa_dift_output/train.json \
    --eval_path ./qa_dift_output/valid.json \
    --test_path ./qa_dift_output/test.json \
    --output_dir ./output/qa_model

# With knowledge injection
python train.py \
    --model_class KGELlama \
    --model_name_or_path /path/to/llama-2-7b-chat-hf \
    --kge_model embeddings \
    --dataset ./qa_dift_output/embeddings \
    --train_path ./qa_dift_output/train.json \
    --eval_path ./qa_dift_output/valid.json \
    --test_path ./qa_dift_output/test.json \
    --output_dir ./output/qa_kge_model
```

## Input Format

Your QA dataset should be a JSON file with this structure:

```json
[
    {
        "question": "Your question here?",
        "answer": "The correct answer",
        "candidates": ["Answer 1", "Answer 2", "Answer 3"],  // optional
        "context": "Additional context..."  // optional
    }
]
```

**Alternative field names:** You can also use `input` instead of `question`, and `output` instead of `answer`.

## All-in-One Script

Use the prepared script to run the complete pipeline:

```bash
bash scripts/prepare_qa_dataset.sh
```

Then follow the on-screen instructions for training.

## Testing

Before processing your own data, test with the example:

```bash
# Run all tests
python validate_qa_adapter.py  # Validation
python test_qa_adapter.py      # Unit tests
python test_e2e_qa_adapter.py  # Integration test
```

## Need Help?

- **Full documentation:** See [QA_ADAPTER_README.md](QA_ADAPTER_README.md)
- **Technical details:** See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **DIFT basics:** See [README.md](README.md)

## Common Issues

### "ModuleNotFoundError: No module named 'transformers'"
Install dependencies:
```bash
pip install transformers torch tqdm
```

### "Tokenizer not found"
Make sure `--llm_dir` points to a valid Hugging Face model directory containing tokenizer files.

### "No output files found"
The conversion hasn't been run yet. Run `qa_data_construct.py` first.

## What Gets Created?

After running the adapter, you'll have:
- `train.json` - Training data in DIFT format
- `valid.json` - Validation data in DIFT format
- `test.json` - Test data in DIFT format
- `stats.json` - Dataset statistics
- `embeddings/` (if created) - Query and entity embeddings

All files are ready to use with the DIFT training pipeline!
