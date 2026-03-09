#!/bin/bash
set -e

# Multi-file joint downsampling for global fairness
DATASET="family"
MAX_RATIO=0.01
QUESTION_DIR_SUFFIX="question"

if [ "$DATASET" = "family" ]; then
    BASE_DIR="data/family"
    QUESTION_BASE=$(find "$BASE_DIR" -name "*${QUESTION_DIR_SUFFIX}" -type d | head -1)
elif [ "$DATASET" = "fb15k-237" ]; then
    BASE_DIR="data/fb15k-237"
    QUESTION_BASE=$(find "$BASE_DIR" -name "*${QUESTION_DIR_SUFFIX}" -type d | head -1)
else
    echo "Error: Unsupported dataset '${DATASET}'"
    exit 1
fi

if [ -z "$QUESTION_BASE" ]; then
    echo "Error: Cannot find question directory with suffix '${QUESTION_DIR_SUFFIX}'"
    exit 1
fi

echo "Multi-file Answer Downsampling"
echo "Dataset: ${DATASET}"
echo "Question directory: ${QUESTION_BASE}"
echo "Max global ratio: ${MAX_RATIO}"

declare -a INPUT_FILES=()

find_split_file() {
    local split=$1
    local base_dir=$2
    local patterns=("${split}_cleaned_final.tsv" "${split}_cleaned.tsv" "${split}.tsv")

    for pattern in "${patterns[@]}"; do
        local file="${base_dir}/${pattern}"
        if [ -f "$file" ]; then
            echo "$file"
            return 0
        fi
    done
    return 1
}

for split in "train" "val" "test"; do
    if file=$(find_split_file "$split" "$QUESTION_BASE"); then
        INPUT_FILES+=("$file")
        echo "Found file: $(basename "$file")"
    fi
done

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "No standard split files found, searching all TSV files..."
    while IFS= read -r -d '' file; do
        INPUT_FILES+=("$file")
        echo "Found file: $(basename "$file")"
    done < <(find "$QUESTION_BASE" -name "*.tsv" -print0)
fi

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "Error: No TSV files found in ${QUESTION_BASE}"
    exit 1
fi

echo "Processing ${#INPUT_FILES[@]} files"

python utils/auto_downsample_answers.py \
    --input "${INPUT_FILES[@]}" \
    --max_ratio "$MAX_RATIO" \
    --answer_column "answer" \
    --id_column "id" \
    --suffix "_downsampled" \
    --random_seed 42

echo "Multi-file downsampling completed"