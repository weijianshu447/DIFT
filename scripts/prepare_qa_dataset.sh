#!/bin/bash
# Example script for adapting QA datasets to DIFT and training

set -e

echo "======================================================================"
echo "QA Dataset Adaptation and Training Pipeline for DIFT"
echo "======================================================================"
echo ""

# Configuration
LLM_DIR="llama-2-7b-chat-hf"  # Change this to your LLM path
QA_INPUT_FILE="example_qa_dataset.json"
QA_OUTPUT_DIR="./qa_dataset_dift"
EMBEDDINGS_DIR="${QA_OUTPUT_DIR}/embeddings"

echo "Step 1: Validate input QA dataset"
echo "----------------------------------------------------------------------"
python validate_qa_adapter.py
echo ""

echo "Step 2: Convert QA dataset to DIFT format"
echo "----------------------------------------------------------------------"
python qa_data_construct.py \
    --llm_dir ${LLM_DIR} \
    --input_file ${QA_INPUT_FILE} \
    --output_dir ${QA_OUTPUT_DIR} \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --add_special_tokens True \
    --add_context True \
    --seed 42

echo ""
echo "Step 3: Validate DIFT-formatted output"
echo "----------------------------------------------------------------------"
# The validation script will now check the output files
python validate_qa_adapter.py
echo ""

echo "Step 4: Prepare dummy embeddings (for KGELlama model)"
echo "----------------------------------------------------------------------"
python qa_utils.py \
    --train_path ${QA_OUTPUT_DIR}/train.json \
    --valid_path ${QA_OUTPUT_DIR}/valid.json \
    --test_path ${QA_OUTPUT_DIR}/test.json \
    --output_dir ${EMBEDDINGS_DIR} \
    --embedding_dim 768

echo ""
echo "======================================================================"
echo "QA Dataset Successfully Prepared!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "Option 1: Train with standard LlamaForCausalLM (no knowledge injection)"
echo "----------------------------------------------------------------------"
echo "python train.py \\"
echo "    --model_class LlamaForCausalLM \\"
echo "    --model_name_or_path ${LLM_DIR} \\"
echo "    --dataset ${QA_OUTPUT_DIR} \\"
echo "    --train_path ${QA_OUTPUT_DIR}/train.json \\"
echo "    --eval_path ${QA_OUTPUT_DIR}/valid.json \\"
echo "    --test_path ${QA_OUTPUT_DIR}/test.json \\"
echo "    --output_dir ./output/qa_model \\"
echo "    --num_train_epochs 3 \\"
echo "    --per_device_train_batch_size 1 \\"
echo "    --gradient_accumulation_steps 16 \\"
echo "    --learning_rate 0.0002 \\"
echo "    --source_max_len 2048 \\"
echo "    --target_max_len 64"
echo ""
echo "Option 2: Train with KGELlama (with knowledge injection)"
echo "----------------------------------------------------------------------"
echo "python train.py \\"
echo "    --model_class KGELlama \\"
echo "    --model_name_or_path ${LLM_DIR} \\"
echo "    --kge_model embeddings \\"
echo "    --dataset ${EMBEDDINGS_DIR} \\"
echo "    --train_path ${QA_OUTPUT_DIR}/train.json \\"
echo "    --eval_path ${QA_OUTPUT_DIR}/valid.json \\"
echo "    --test_path ${QA_OUTPUT_DIR}/test.json \\"
echo "    --output_dir ./output/qa_kge_model \\"
echo "    --num_train_epochs 3 \\"
echo "    --per_device_train_batch_size 1 \\"
echo "    --gradient_accumulation_steps 16 \\"
echo "    --learning_rate 0.0002 \\"
echo "    --embedding_dim 768 \\"
echo "    --source_max_len 2048 \\"
echo "    --target_max_len 64"
echo ""
echo "======================================================================"
