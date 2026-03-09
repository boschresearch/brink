#!/bin/bash
set -e

# Single file grounding generation
BASE_DIR="data/family"
FACTS_PATH="$BASE_DIR/facts.tsv"
RULES_PATH="$BASE_DIR/rules_with_global_index.json"
OUTPUT_DIR="$BASE_DIR/grounding_output"

mkdir -p "$OUTPUT_DIR"

echo "Processing rule file: $RULES_PATH"
echo "Facts file: $FACTS_PATH"
echo "Output directory: $OUTPUT_DIR"

if [ -n "$MAX_RULES" ] && [ "$MAX_RULES" -gt 0 ]; then
    echo "Max rules to process: $MAX_RULES"
else
    echo "Processing all rules"
fi

python grounding_generation.py \
  --facts "$FACTS_PATH" \
  --rules "$RULES_PATH" \
  --output "$OUTPUT_DIR" \
  ${MAX_RULES:+--max_rules "$MAX_RULES"}

echo "Grounding generation completed"
echo "Output saved to: $OUTPUT_DIR"