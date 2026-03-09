#!/bin/bash
set -e

# Convert AMIE output and assign global indices
DATASET="family"
MAXAD="3"

BASE_DIR="data/${DATASET}"
AMIE_TXT_FILE="amie_output_family.txt"
TEMP_JSON_FILE="${BASE_DIR}/rules_string.json"
FINAL_JSON_FILE="${BASE_DIR}/rules_with_global_index.json"

echo "Converting and indexing AMIE rules"
echo "Dataset: ${DATASET}"
echo "Max atoms: ${MAXAD}"

# Step 1: Convert AMIE output to JSON
echo "Step 1: Converting AMIE txt to JSON..."
python convert_amie_txt_to_json.py "${AMIE_TXT_FILE}" "${TEMP_JSON_FILE}"

# Step 2: Assign global indices
echo "Step 2: Assigning global rule indices..."
python utils/assign_global_rule_index.py \
  -i "${TEMP_JSON_FILE}" \
  -o "${FINAL_JSON_FILE}"

# Step 3: Clean up temporary files
echo "Step 3: Cleaning up temporary files..."
rm "${TEMP_JSON_FILE}"

echo "Process completed successfully"
echo "Final output: ${FINAL_JSON_FILE}"