#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="data/family/grounding_output/grounding_splits/val.tsv"
OUTPUT_FILE="data/family/question/val.tsv"
N_INSTANCES=220
MODEL_CHOICE="openai"
PROMPT_TYPE="general"

python3 question_generation.py \
  --input_file  "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --n_instances "$N_INSTANCES" \
  --model_choice "$MODEL_CHOICE" \
  --prompt_type  "$PROMPT_TYPE"