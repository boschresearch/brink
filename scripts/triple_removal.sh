#!/bin/bash
set -e

# Generate incomplete knowledge graph by removing head triples
DATASET="family"
N_INSTANCES=30

if [ "$DATASET" = "family" ]; then
    BASE_DIR="data/family"
    KG_FILE="${BASE_DIR}/facts.tsv"
    GROUNDING_FILE="${BASE_DIR}/grounding_output/filtered_grounding_output_${N_INSTANCES}_instance_before_question_generation.tsv"
    OUTPUT_FILE="${BASE_DIR}/grounding_output/grounding_splits/incomplete_facts_string_of_${N_INSTANCES}_instance.tsv"
elif [ "$DATASET" = "fb15k-237" ]; then
    BASE_DIR="data/fb15k-237"
    KG_FILE="${BASE_DIR}/facts.tsv"
    GROUNDING_FILE="${BASE_DIR}/grounding_output/filtered_grounding_output_${N_INSTANCES}_instance_before_question_generation.tsv"
    OUTPUT_FILE="${BASE_DIR}/grounding_output/grounding_splits/incomplete_facts_string_of_${N_INSTANCES}_instance.tsv"
else
    echo "Error: Unsupported dataset '${DATASET}'. Use 'family' or 'fb15k-237'."
    exit 1
fi

echo "Generating Incomplete Knowledge Graph"
echo "Dataset: ${DATASET}"
echo "Grounding instances: ${N_INSTANCES}"

if [ ! -f "${KG_FILE}" ]; then
    echo "Error: KG file not found: ${KG_FILE}"
    exit 1
fi

if [ ! -f "${GROUNDING_FILE}" ]; then
    echo "Error: Grounding file not found: ${GROUNDING_FILE}"
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"

echo "Input KG file: ${KG_FILE}"
echo "Grounding file: ${GROUNDING_FILE}"
echo "Output file: ${OUTPUT_FILE}"

echo "Generating incomplete KG by removing head triples..."
python incomplete_kg_generation.py \
  --kg_file "${KG_FILE}" \
  --grounding_file "${GROUNDING_FILE}" \
  --output_file "${OUTPUT_FILE}"

echo "Incomplete KG generation completed"
echo "Incomplete KG saved to: ${OUTPUT_FILE}"